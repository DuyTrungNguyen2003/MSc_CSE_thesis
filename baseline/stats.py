import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats

data = {
    # Unsampled
    '~25K genes': [0.8487, 0.9387, 0.7401, 0.7138, 0.8116],
    'L1 regularization': [0.9279, 0.9704, 0.9404, 0.8846, 0.8336],
    '49 signature genes': [0.8671, 0.9546, 0.9002, 0.9343, 0.8403],
    '157 signature genes': [0.8976, 0.9619, 0.8434, 0.8872, 0.8640],
    # Sampled
    '~25K genes (Sampled)': [0.9232, 0.9871, 0.8848, 0.8514, 0.8369],
    'L1 regularization (Sampled)': [0.9204, 0.9757, 0.9166, 0.9177, 0.9329],
    '49 signature genes (Sampled)': [0.8870, 0.9666, 0.9042, 0.9494, 0.8580],
    '157 signature genes (Sampled)': [0.9352, 0.9510, 0.9176, 0.9510, 0.8992]
}

order_list = list(data.keys()) 

df = pd.DataFrame(data)
df_melted = df.melt(var_name='Method', value_name='AUROC')
df_melted['Condition'] = df_melted['Method'].apply(lambda x: 'Sampled' if '(Sampled)' in x else 'Unsampled')


# stats
stats_df = []
for col in data:
    vals = data[col]
    n = len(vals)
    mean = np.mean(vals)
    std_dev = np.std(vals, ddof=1)
    std_err = stats.sem(vals)
    ci_95 = std_err * stats.t.ppf((1 + 0.95) / 2., n-1)
    
    stats_df.append({
        'Method': col,
        'Mean': mean,
        'SD': std_dev,
        'CI_95': ci_95
    })

stats_df = pd.DataFrame(stats_df)


# plot
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

sns.swarmplot(
    data=df_melted, 
    x='Method', 
    y='AUROC', 
    hue='Condition', 
    order=order_list,
    dodge=True,
    size=8, 
    palette={'Unsampled': "royalblue", 'Sampled': "forestgreen"}
)

plt.errorbar(
    x=stats_df['Method'], 
    y=stats_df['Mean'], 
    yerr=stats_df['CI_95'], 
    fmt='o',            
    color='black',      
    ecolor='black',     
    capsize=5,          
    elinewidth=2,       
    label='Mean ± 95% CI'
)

plt.title('Performance Baselines', fontsize=16)
plt.ylabel('AUROC', fontsize=14)
plt.xlabel('Method', fontsize=14)
plt.ylim(0.65, 1) 
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.axvline(x=3.5, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('/project_antwerp/baseline/averaging_baseline/performance_baselines.png', dpi=300, bbox_inches='tight')