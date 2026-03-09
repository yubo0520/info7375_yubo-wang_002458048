# Assignment 1: E-Commerce Customer Retention with MDP

Using MDP (Markov Decision Process) to figure out a real problem: should an e-commerce platform send coupons to users to prevent churn.

## The Problem

An e-commerce company has a bunch of users — some are active, some are about to leave. Sending coupons can help retain users, but each coupon costs money (assume $20). The question is: **which users are worth sending coupons to, and which ones are just a waste of money?**

## How I Modeled It

Framed this as an MDP:

- **State space**: User tenure (New / Mid / Loyal) × recent activity (Active / Neutral / Inactive) = 9 user states
- **Action space**: Two choices — Do Nothing or Send Coupon
- **Transition probabilities**: Calculated from the dataset — churn probability for each state-action pair
- **Rewards**: Retaining a user = +$100, losing a user = -$100, sending a coupon costs an extra $20

Then ran **Policy Iteration** to find the optimal policy.

## Data

Used an e-commerce dataset (`E_Commerce_Dataset.csv`, 5630 records) with fields like user tenure, days since last order, coupon usage, churn label, etc. Missing values filled with median.

## Results

Converged in just 2 iterations. The optimal policy:

| User State | Optimal Policy | Estimated Value (LTV) |
|-----------|---------------|----------------------|
| Loyal_Active | Do Nothing | 727.72 |
| Loyal_Inactive | Do Nothing | 782.89 |
| Loyal_Neutral | Do Nothing | 820.65 |
| Mid_Active | Do Nothing | 258.87 |
| Mid_Inactive | **Send Coupon** | 565.95 |
| Mid_Neutral | **Send Coupon** | 419.91 |
| New_Active | Do Nothing | 56.46 |
| New_Inactive | Do Nothing | 161.02 |
| New_Neutral | Do Nothing | 187.07 |

Basically: only **Mid-tenure users who are not very active** are worth sending coupons to. Loyal users are already stable, and New users don't have enough value to justify the cost. Makes sense.

## How to Run

Just open the notebook:

```
jupyter notebook ecommerce_mdp.ipynb
```

## Files

- `ecommerce_mdp.ipynb` — Main code: data processing, MDP modeling, policy iteration, results
- `E_Commerce_Dataset.csv` — Raw dataset
- `mdp_policy_results.csv` — Output: optimal policy results

## Presentation Video

[Watch the recording here](https://drive.google.com/file/d/1UgO1A_Xjlu53ukcjTzAkDLYGkiP1udqz/view?usp=sharing)
