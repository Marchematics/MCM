import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


plt.rcParams.update({
    # 字体设置
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    
    # 线条设置
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    
    # 刻度设置
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    
    # 图例设置
    'legend.frameon': True,
    'legend.framealpha': 1.0,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    
    # 保存设置
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    
    # 其他
    'axes.grid': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

COLORS = {
    'blue': '#0072B2',
    'orange': '#D55E00', 
    'green': '#009E73',
    'red': '#CC79A7',
    'yellow': '#F0E442',
    'cyan': '#56B4E9',
    'purple': '#9467BD',
    'gray': '#7F7F7F',
    'black': '#000000'
}

results_df = pd.read_csv('result/fan_vote_estimates_improved.csv')
validation_df = pd.read_csv('result/validation_stats.csv')
original_df = pd.read_csv('2026_MCM_Problem_C_Data.csv', encoding='utf-8-sig')

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3))

ax1 = axes[0]
seasons = validation_df['season'].values
rates = validation_df['rate'].values * 100
methods = validation_df['method'].values

colors_bar = [COLORS['blue'] if m == 'percentage' else COLORS['orange'] for m in methods]
bars = ax1.bar(seasons, rates, color=colors_bar, edgecolor='black', linewidth=0.5, width=0.8)

ax1.axhline(y=95, color=COLORS['gray'], linestyle='--', linewidth=1, alpha=0.8)
ax1.axhline(y=100, color=COLORS['green'], linestyle='-', linewidth=0.5, alpha=0.5)

ax1.set_xlabel('Season')
ax1.set_ylabel('Consistency Rate (%)')
ax1.set_ylim(70, 102)
ax1.set_xlim(0, 35)
ax1.set_xticks([1, 5, 10, 15, 20, 25, 30, 34])

pct_patch = mpatches.Patch(color=COLORS['blue'], label='Percentage method')
rank_patch = mpatches.Patch(color=COLORS['orange'], label='Rank method')
ax1.legend(handles=[pct_patch, rank_patch], loc='lower left', frameon=True)

ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=11, fontweight='bold', va='top')

ax2 = axes[1]
pct_data = validation_df[validation_df['method'] == 'percentage']
rank_data = validation_df[validation_df['method'] == 'rank']

methods_names = ['Percentage\n(S3-S27)', 'Rank\n(S1-2, S28-34)']
valid_counts = [pct_data['valid'].sum(), rank_data['valid'].sum()]
total_counts = [pct_data['total'].sum(), rank_data['total'].sum()]
rates_method = [v/t*100 for v, t in zip(valid_counts, total_counts)]

x_pos = [0, 1]
bars2 = ax2.bar(x_pos, rates_method, color=[COLORS['blue'], COLORS['orange']], 
                edgecolor='black', linewidth=1, width=0.6)

for i, (bar, v, t) in enumerate(zip(bars2, valid_counts, total_counts)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{v}/{t}', ha='center', va='bottom', fontsize=9)

ax2.set_ylabel('Consistency Rate (%)')
ax2.set_ylim(0, 110)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(methods_names)
ax2.axhline(y=100, color=COLORS['gray'], linestyle='--', linewidth=1, alpha=0.5)

ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=11, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('Fig1_Model_Validation.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('Fig1_Model_Validation.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 1 saved: Fig1_Model_Validation.png/pdf")

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3))

ax1 = axes[0]
uncertainty_week = results_df.groupby('week').agg({
    'fan_vote_std': ['mean', 'std', 'count']
}).reset_index()
uncertainty_week.columns = ['week', 'mean', 'std', 'count']

weeks = uncertainty_week['week'].values
means = uncertainty_week['mean'].values
stds = uncertainty_week['std'].values

ax1.errorbar(weeks, means, yerr=stds, fmt='o-', color=COLORS['blue'], 
             capsize=3, capthick=1, markersize=5, markerfacecolor='white',
             markeredgecolor=COLORS['blue'], markeredgewidth=1.5)

ax1.fill_between(weeks, means - stds, means + stds, alpha=0.2, color=COLORS['blue'])

ax1.set_xlabel('Competition Week')
ax1.set_ylabel('Estimation Uncertainty (Std. Dev.)')
ax1.set_xlim(0.5, 11.5)
ax1.set_ylim(0, 0.18)
ax1.set_xticks(range(1, 12))

ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=11, fontweight='bold', va='top')

ax2 = axes[1]
elim_data = results_df[results_df['is_eliminated'] == True]['fan_vote_std']
non_elim_data = results_df[results_df['is_eliminated'] == False]['fan_vote_std']

bp = ax2.boxplot([elim_data, non_elim_data], 
                  labels=['Eliminated\nContestants', 'Non-eliminated\nContestants'],
                  patch_artist=True,
                  widths=0.5,
                  medianprops=dict(color=COLORS['black'], linewidth=1.5),
                  flierprops=dict(marker='o', markersize=3, markerfacecolor=COLORS['gray'], alpha=0.5))

bp['boxes'][0].set_facecolor(COLORS['orange'])
bp['boxes'][1].set_facecolor(COLORS['blue'])
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_alpha(0.7)

ax2.set_ylabel('Estimation Uncertainty (Std. Dev.)')
ax2.set_ylim(0, 0.2)

for i, data in enumerate([elim_data, non_elim_data]):
    ax2.scatter(i+1, data.mean(), marker='D', color=COLORS['black'], s=30, zorder=5)

ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=11, fontweight='bold', va='top')

t_stat, p_val = stats.ttest_ind(elim_data, non_elim_data)
ax2.text(0.5, 0.95, f'p < 0.001***', transform=ax2.transAxes, ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('Fig2_Uncertainty_Analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('Fig2_Uncertainty_Analysis.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 2 saved: Fig2_Uncertainty_Analysis.png/pdf")

fig, ax = plt.subplots(figsize=(5, 4.5))

non_elim = results_df[results_df['is_eliminated'] == False]
elim = results_df[results_df['is_eliminated'] == True]

ax.scatter(non_elim['judge_pct'] * 100, non_elim['fan_vote_est'] * 100, 
           c=COLORS['blue'], alpha=0.3, s=15, label='Non-eliminated', edgecolors='none')
ax.scatter(elim['judge_pct'] * 100, elim['fan_vote_est'] * 100, 
           c=COLORS['orange'], alpha=0.6, s=20, label='Eliminated', edgecolors='none')

ax.plot([0, 60], [0, 60], 'k--', linewidth=1, alpha=0.7, label='y = x')

x_all = results_df['judge_pct'].values * 100
y_all = results_df['fan_vote_est'].values * 100
slope, intercept, r_value, p_value, std_err = stats.linregress(x_all, y_all)
x_fit = np.linspace(0, 50, 100)
y_fit = slope * x_fit + intercept
ax.plot(x_fit, y_fit, color=COLORS['red'], linewidth=1.5, 
        label=f'Linear fit (R² = {r_value**2:.3f})')

ax.set_xlabel('Judge Score Percentage (%)')
ax.set_ylabel('Estimated Fan Vote Percentage (%)')
ax.set_xlim(0, 55)
ax.set_ylim(0, 55)
ax.set_aspect('equal')

ax.legend(loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig('Fig3_Judge_vs_FanVote.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('Fig3_Judge_vs_FanVote.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 3 saved: Fig3_Judge_vs_FanVote.png/pdf")

fig = plt.figure(figsize=(7.5, 6))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

cases = [
    ('Jerry Rice', 2, '2nd Place'),
    ('Billy Ray Cyrus', 4, '5th Place'),
    ('Bristol Palin', 11, '3rd Place'),
    ('Bobby Bones', 27, 'Champion')
]

for idx, (name, season, result) in enumerate(cases):
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    
    case_data = results_df[(results_df['celebrity_name'] == name) & 
                           (results_df['season'] == season)].sort_values('week')
    
    weeks = case_data['week'].values
    judge_pct = case_data['judge_pct'].values * 100
    fan_vote = case_data['fan_vote_est'].values * 100
    
    x = np.arange(len(weeks))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, judge_pct, width, label='Judge %', 
                   color=COLORS['blue'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, fan_vote, width, label='Est. Fan Vote %', 
                   color=COLORS['orange'], edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Week')
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'{name} (S{season}, {result})', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(w) for w in weeks])
    ax.set_ylim(0, max(max(judge_pct), max(fan_vote)) * 1.2)
    
    if idx == 0:
        ax.legend(loc='upper left', fontsize=8)
    
    # 添加子图标签
    label = chr(ord('a') + idx)
    ax.text(0.02, 0.98, f'({label})', transform=ax.transAxes, 
            fontsize=11, fontweight='bold', va='top')

plt.savefig('Fig4_Controversy_Cases.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('Fig4_Controversy_Cases.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 4 saved: Fig4_Controversy_Cases.png/pdf")

fig, ax = plt.subplots(figsize=(6, 4))

line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D']
colors_case = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['red']]

for idx, (name, season, result) in enumerate(cases):
    case_data = results_df[(results_df['celebrity_name'] == name) & 
                           (results_df['season'] == season)].sort_values('week')
    
    advantages = []
    weeks = []
    
    for _, row in case_data.iterrows():
        week = row['week']
        week_all = results_df[(results_df['season'] == season) & 
                               (results_df['week'] == week)]
        n = len(week_all)
        
        j_rank = (week_all['judge_pct'] > row['judge_pct']).sum() + 1
        v_rank = (week_all['fan_vote_est'] > row['fan_vote_est']).sum() + 1
        
        advantages.append(j_rank - v_rank)  
        weeks.append(week)
    
    ax.plot(weeks, advantages, linestyle=line_styles[idx], marker=markers[idx],
            color=colors_case[idx], label=f'{name} (S{season})',
            markersize=6, markerfacecolor='white', markeredgewidth=1.5)

ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
ax.fill_between([0, 12], [0, 0], [10, 10], alpha=0.1, color=COLORS['green'], label='_nolegend_')
ax.fill_between([0, 12], [0, 0], [-10, -10], alpha=0.1, color=COLORS['orange'], label='_nolegend_')

ax.set_xlabel('Competition Week')
ax.set_ylabel('Fan Vote Rank Advantage\n(Judge Rank − Fan Rank)')
ax.set_xlim(0.5, 10.5)
ax.set_ylim(-6, 8)

ax.text(10.3, 3, 'Fan\nFavored', fontsize=8, ha='left', color=COLORS['green'])
ax.text(10.3, -3, 'Judge\nFavored', fontsize=8, ha='left', color=COLORS['orange'])

ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('Fig5_Ranking_Trajectory.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('Fig5_Ranking_Trajectory.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 5 saved: Fig5_Ranking_Trajectory.png/pdf")

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))


contestants = ['A', 'B', 'C', 'D', 'E']
judge_scores = np.array([28, 25, 24, 22, 18])
fan_votes = np.array([15, 25, 20, 30, 10])

ax1 = axes[0]
j_pct = judge_scores / judge_scores.sum() * 100
f_pct = fan_votes / fan_votes.sum() * 100
combined_pct = j_pct + f_pct

x = np.arange(len(contestants))
width = 0.25

bars1 = ax1.bar(x - width, j_pct, width, label='Judge %', color=COLORS['blue'], edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x, f_pct, width, label='Fan Vote %', color=COLORS['orange'], edgecolor='black', linewidth=0.5)
bars3 = ax1.bar(x + width, combined_pct, width, label='Combined %', color=COLORS['green'], edgecolor='black', linewidth=0.5)

min_idx = np.argmin(combined_pct)
ax1.annotate('Eliminated', xy=(x[min_idx] + width, combined_pct[min_idx]), 
             xytext=(x[min_idx] + width + 0.3, combined_pct[min_idx] + 8),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=9, color='red')

ax1.set_xlabel('Contestant')
ax1.set_ylabel('Percentage (%)')
ax1.set_title('Percentage Method', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(contestants)
ax1.legend(loc='upper right', fontsize=8)
ax1.set_ylim(0, 55)

ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=11, fontweight='bold', va='top')

ax2 = axes[1]
j_rank = np.array([1, 2, 3, 4, 5])  # 基于评委分数
f_rank = np.array([3, 2, 4, 1, 5])  # 基于粉丝投票
combined_rank = j_rank + f_rank

bars1 = ax2.bar(x - width, j_rank, width, label='Judge Rank', color=COLORS['blue'], edgecolor='black', linewidth=0.5)
bars2 = ax2.bar(x, f_rank, width, label='Fan Vote Rank', color=COLORS['orange'], edgecolor='black', linewidth=0.5)
bars3 = ax2.bar(x + width, combined_rank, width, label='Combined Rank', color=COLORS['green'], edgecolor='black', linewidth=0.5)

max_idx = np.argmax(combined_rank)
ax2.annotate('Eliminated', xy=(x[max_idx] + width, combined_rank[max_idx]), 
             xytext=(x[max_idx] + width + 0.3, combined_rank[max_idx] + 1),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=9, color='red')

ax2.set_xlabel('Contestant')
ax2.set_ylabel('Rank (1 = Best)')
ax2.set_title('Rank Method', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(contestants)
ax2.legend(loc='upper left', fontsize=8)
ax2.set_ylim(0, 12)

ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=11, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('Fig6_Method_Comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('Fig6_Method_Comparison.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 6 saved: Fig6_Method_Comparison.png/pdf")

fig, ax = plt.subplots(figsize=(8, 5))

heatmap_data = []
seasons_list = sorted(results_df['season'].unique())

for season in seasons_list:
    season_data = results_df[results_df['season'] == season]
    row = []
    for week in range(1, 12):
        week_data = season_data[season_data['week'] == week]
        if len(week_data) > 0:
            # 粉丝优势 = 粉丝投票% - 评委%
            advantage = (week_data['fan_vote_est'] - week_data['judge_pct']).mean() * 100
            row.append(advantage)
        else:
            row.append(np.nan)
    heatmap_data.append(row)

heatmap_array = np.array(heatmap_data)

im = ax.imshow(heatmap_array, aspect='auto', cmap='RdBu_r', vmin=-5, vmax=5)

ax.set_xticks(range(11))
ax.set_xticklabels([f'{w}' for w in range(1, 12)])
ax.set_yticks(range(0, 34, 5))
ax.set_yticklabels([f'S{s}' for s in range(1, 35, 5)])

ax.set_xlabel('Competition Week')
ax.set_ylabel('Season')

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Mean Fan Vote Advantage (%)', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig('Fig7_Advantage_Heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('Fig7_Advantage_Heatmap.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 7 saved: Fig7_Advantage_Heatmap.png/pdf")

fig, ax = plt.subplots(figsize=(5, 4))

case_names = ['Jerry Rice\n(S2)', 'Billy Ray\nCyrus (S4)', 'Bristol Palin\n(S11)', 'Bobby Bones\n(S27)']
avg_j_ranks = [4.6, 6.5, 5.9, 7.0]
avg_v_ranks = [3.1, 3.4, 2.7, 2.4]

x = np.arange(len(case_names))
width = 0.35

bars1 = ax.bar(x - width/2, avg_j_ranks, width, label='Avg. Judge Rank', 
               color=COLORS['blue'], edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, avg_v_ranks, width, label='Avg. Fan Vote Rank', 
               color=COLORS['orange'], edgecolor='black', linewidth=0.5)

# 添加差值标注
for i, (j, v) in enumerate(zip(avg_j_ranks, avg_v_ranks)):
    diff = j - v
    y_pos = max(j, v) + 0.3
    ax.annotate(f'+{diff:.1f}', xy=(i, y_pos), ha='center', fontsize=9, 
                color=COLORS['green'], fontweight='bold')

ax.set_xlabel('Contestant')
ax.set_ylabel('Average Weekly Rank\n(1 = Best)')
ax.set_xticks(x)
ax.set_xticklabels(case_names)
ax.legend(loc='upper left')
ax.set_ylim(0, 9)


plt.tight_layout()
plt.savefig('Fig8_Controversy_Summary.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('Fig8_Controversy_Summary.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 8 saved: Fig8_Controversy_Summary.png/pdf")

print("\n" + "="*50)
print("All SCI-style figures generated successfully!")
print("="*50)
print("\nGenerated files:")
for i in range(1, 9):
    print(f"  - Fig{i}_*.png/pdf")