import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Create plots directory if it doesn't exist
base_dir = r'c:\Users\victus\Desktop\Online Shopper Purchase Intention Prediction using an End-to-End ML Pipeline'
data_path = os.path.join(base_dir, 'data', 'online_shoppers_intention.csv')
plots_dir = os.path.join(base_dir, 'data', 'plots')
os.makedirs(plots_dir, exist_ok=True)

print("="*70)
print("EXECUTING EDA NOTEBOOK - GENERATING VISUALIZATIONS")
print("="*70)

# Load dataset
print("\n1. Loading dataset...")
df = pd.read_csv(data_path)
print(f"   ✅ Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Missing values check
print("\n2. Checking for missing values...")
missing_total = df.isnull().sum().sum()
print(f"   ✅ Total missing values: {missing_total}")

# Class distribution
print("\n3. Analyzing class distribution...")
revenue_counts = df['Revenue'].value_counts()
revenue_percentages = df['Revenue'].value_counts(normalize=True) * 100
print(f"   - No Purchase: {revenue_counts[False]:,} ({revenue_percentages[False]:.2f}%)")
print(f"   - Purchase: {revenue_counts[True]:,} ({revenue_percentages[True]:.2f}%)")

# Plot 1: Class Distribution
print("\n4. Generating Plot 1: Class Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#e74c3c', '#2ecc71']

revenue_counts.plot(kind='bar', ax=axes[0], color=colors, edgecolor='black', alpha=0.8)
axes[0].set_title('Revenue Distribution (Count)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Revenue', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_xticklabels(['No Purchase', 'Purchase'], rotation=0)
axes[0].grid(axis='y', alpha=0.3)

for i, v in enumerate(revenue_counts.values):
    axes[0].text(i, v + 200, f'{v:,}', ha='center', va='bottom', fontweight='bold')

axes[1].pie(revenue_counts.values, labels=['No Purchase', 'Purchase'], autopct='%1.1f%%',
            colors=colors, startangle=90, explode=(0.05, 0.05), shadow=True)
axes[1].set_title('Revenue Distribution (Percentage)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '01_class_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 01_class_distribution.png")

# Plot 2: Purchase by Visitor Type
print("\n5. Generating Plot 2: Purchase by Visitor Type...")
visitor_revenue = pd.crosstab(df['VisitorType'], df['Revenue'], normalize='index') * 100

fig, ax = plt.subplots(figsize=(12, 6))
visitor_revenue.plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'], 
                      edgecolor='black', alpha=0.8)
ax.set_title('Purchase Rate by Visitor Type', fontsize=14, fontweight='bold')
ax.set_xlabel('Visitor Type', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_xticklabels(visitor_revenue.index, rotation=45, ha='right')
ax.legend(['No Purchase', 'Purchase'], title='Revenue')
ax.grid(axis='y', alpha=0.3)

for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', padding=3)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '02_purchase_by_visitor_type.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 02_purchase_by_visitor_type.png")

# Plot 3: Purchase by Month
print("\n6. Generating Plot 3: Purchase by Month...")
month_revenue = pd.crosstab(df['Month'], df['Revenue'], normalize='index') * 100
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_revenue = month_revenue.reindex([m for m in month_order if m in month_revenue.index])

fig, ax = plt.subplots(figsize=(14, 6))
month_revenue[True].plot(kind='bar', ax=ax, color='#3498db', edgecolor='black', alpha=0.8)
ax.set_title('Purchase Rate by Month', fontsize=14, fontweight='bold')
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Purchase Rate (%)', fontsize=12)
ax.set_xticklabels(month_revenue.index, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

for i, v in enumerate(month_revenue[True].values):
    ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

avg_purchase_rate = month_revenue[True].mean()
ax.axhline(y=avg_purchase_rate, color='red', linestyle='--', linewidth=2, 
           label=f'Average: {avg_purchase_rate:.1f}%')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '03_purchase_by_month.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 03_purchase_by_month.png")

# Plot 4: Numerical Distributions
print("\n7. Generating Plot 4: Numerical Feature Distributions...")
key_features = ['PageValues', 'BounceRates', 'ExitRates']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, feature in enumerate(key_features):
    axes[i].hist(df[feature], bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[i].set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel(feature, fontsize=10)
    axes[i].set_ylabel('Frequency', fontsize=10)
    axes[i].grid(axis='y', alpha=0.3)
    
    mean_val = df[feature].mean()
    median_val = df[feature].median()
    axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    axes[i].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    axes[i].legend()

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '04_numerical_distributions.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 04_numerical_distributions.png")

# Plot 5: Boxplots by Revenue
print("\n8. Generating Plot 5: Boxplots by Revenue...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, feature in enumerate(key_features):
    df.boxplot(column=feature, by='Revenue', ax=axes[i], 
               patch_artist=True, 
               boxprops=dict(facecolor='lightblue', alpha=0.7),
               medianprops=dict(color='red', linewidth=2))
    axes[i].set_title(f'{feature} by Revenue', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Revenue', fontsize=10)
    axes[i].set_ylabel(feature, fontsize=10)
    axes[i].set_xticklabels(['No Purchase', 'Purchase'])
    axes[i].grid(axis='y', alpha=0.3)

plt.suptitle('')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '05_boxplots_by_revenue.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 05_boxplots_by_revenue.png")

# Plot 6: Correlation Heatmap
print("\n9. Generating Plot 6: Correlation Heatmap...")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df_corr = df.copy()
df_corr['Revenue'] = df_corr['Revenue'].astype(int)
correlation_matrix = df_corr[numerical_cols + ['Revenue']].corr()

plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap - Focus on Revenue', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '06_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 06_correlation_heatmap.png")

# Plot 7: Top Correlations
print("\n10. Generating Plot 7: Top Correlations with Revenue...")
revenue_correlations = correlation_matrix['Revenue'].sort_values(ascending=False)
top_n = 10
top_correlations = revenue_correlations[1:top_n+1]

plt.figure(figsize=(12, 6))
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_correlations.values]
top_correlations.plot(kind='barh', color=colors, edgecolor='black', alpha=0.8)
plt.title(f'Top {top_n} Features Correlated with Revenue', fontsize=14, fontweight='bold')
plt.xlabel('Correlation Coefficient', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)

for i, v in enumerate(top_correlations.values):
    plt.text(v + 0.01 if v > 0 else v - 0.01, i, f'{v:.3f}', 
             va='center', ha='left' if v > 0 else 'right', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '07_top_correlations.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 07_top_correlations.png")

# Summary
print("\n" + "="*70)
print("KEY INSIGHTS SUMMARY")
print("="*70)
print(f"\n1. Dataset: {df.shape[0]:,} records, {df.shape[1]} features")
print(f"2. Missing Values: {missing_total}")
print(f"3. Purchase Rate: {revenue_percentages[True]:.2f}% (CLASS IMBALANCE!)")
print(f"4. Visitor Types: {', '.join(df['VisitorType'].unique())}")
print(f"5. Top Correlated Feature: {top_correlations.index[0]} ({top_correlations.values[0]:.3f})")

print("\n" + "="*70)
print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print(f"✅ Saved 7 plots to: data/plots/")
print("="*70)
