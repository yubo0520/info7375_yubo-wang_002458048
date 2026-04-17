import hydra
import ray

from .dataset import AgentDataset
from .trainer import AgentFlowTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.main_ppo import create_rl_sampler


def _disable_flash_attention_if_needed(config):
    """Force a safe attention backend when flash-attn is unavailable."""
    try:
        import flash_attn  # noqa: F401
        return
    except Exception:
        pass

    from omegaconf import open_dict

    with open_dict(config):
        if "actor_rollout_ref" not in config or "model" not in config.actor_rollout_ref:
            return
        model_cfg = config.actor_rollout_ref.model
        override_cfg = model_cfg.get("override_config")
        if override_cfg is None:
            model_cfg.override_config = {}
            override_cfg = model_cfg.override_config

        # Keep both keys for compatibility with different transformers/verl versions.
        override_cfg["_attn_implementation"] = "sdpa"
        override_cfg["attn_implementation"] = "sdpa"


def _load_reward_manager_compat(config, tokenizer, num_examine, reward_kwargs):
    """Load reward manager across verl versions with/without num_examine support."""
    try:
        return load_reward_manager(config, tokenizer, num_examine=num_examine, **reward_kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'num_examine'" not in str(exc):
            raise
        return load_reward_manager(config, tokenizer, **reward_kwargs)


def _build_trainer_compat(*, trainer_cls, common_kwargs, reward_fn, val_reward_fn):
    """Construct trainer across verl API versions."""
    variants = (
        {"reward_fn": reward_fn, "val_reward_fn": val_reward_fn},
        {"reward_function": reward_fn, "val_reward_function": val_reward_fn},
        {},
    )
    last_exc = None
    for reward_kwargs in variants:
        try:
            trainer = trainer_cls(**common_kwargs, **reward_kwargs)
            trainer.reward_fn = reward_fn
            trainer.val_reward_fn = val_reward_fn
            return trainer
        except TypeError as exc:
            msg = str(exc)
            if "unexpected keyword argument" not in msg:
                raise
            last_exc = exc
            continue

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Failed to build trainer due to unknown constructor compatibility issue.")


@hydra.main(config_path="pkg://agentflow/verl", config_name="config", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}
            },
            # num_cpus omitted — ray_init not in config struct
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)
        _disable_flash_attention_if_needed(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_kwargs = config.reward_model.get("reward_kwargs", {})
        reward_fn = _load_reward_manager_compat(config, tokenizer, num_examine=0, reward_kwargs=reward_kwargs)
        val_reward_fn = _load_reward_manager_compat(config, tokenizer, num_examine=1, reward_kwargs=reward_kwargs)
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Use our special dataset
        train_dataset = AgentDataset(
            data_files=config.data.train_files,
            tokenizer=tokenizer,
            processor=processor,
            config=config.data,
        )
        val_dataset = AgentDataset(
            data_files=config.data.val_files,
            tokenizer=tokenizer,
            processor=processor,
            config=config.data,
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer_common_kwargs = dict(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer = _build_trainer_compat(
            trainer_cls=AgentFlowTrainer,
            common_kwargs=trainer_common_kwargs,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
