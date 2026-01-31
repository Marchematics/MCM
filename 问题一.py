import pandas as pd
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.stats import spearmanr, dirichlet, rankdata
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    score_cols = [col for col in df.columns if 'judge' in col]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col].replace('N/A', np.nan), errors='coerce')
    return df

def get_elimination_week(result_str):
    if pd.isna(result_str):
        return None
    if 'Eliminated Week' in str(result_str):
        try:
            return int(result_str.split('Week')[1].strip())
        except:
            return None
    return None

def get_season_data(df, season):
    season_df = df[df['season'] == season].copy()
    
    if season in [1, 2] or season >= 28:
        method = 'rank'
    else:
        method = 'percentage'
    
    weeks_data = {}
    max_week = 1
    
    for week in range(1, 12):
        judge_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
        valid_cols = [col for col in judge_cols if col in season_df.columns]
        
        if not valid_cols:
            continue
        
        week_scores = season_df[valid_cols].sum(axis=1, skipna=True)
        
        active_mask = week_scores > 0
        if active_mask.sum() == 0:
            continue
        
        active_names = season_df.loc[active_mask, 'celebrity_name'].values
        active_scores = week_scores[active_mask].values
        
        eliminated = None
        for idx in season_df[active_mask].index:
            result = season_df.loc[idx, 'results']
            if f'Eliminated Week {week}' in str(result):
                eliminated = season_df.loc[idx, 'celebrity_name']
                break
        
        weeks_data[week] = {
            'names': active_names,
            'judge_scores': active_scores,
            'eliminated': eliminated,
            'n_contestants': len(active_names)
        }
        max_week = week
    
    final_placements = {}
    for idx, row in season_df.iterrows():
        final_placements[row['celebrity_name']] = row['placement']
    
    return {
        'season': season,
        'method': method,
        'weeks': weeks_data,
        'max_week': max_week,
        'final_placements': final_placements,
        'contestants': season_df['celebrity_name'].values
    }



class ImprovedFanVoteEstimator:

    def __init__(self):
        self.results = []
    
    def estimate_week_percentage(self, judge_scores, eliminated_idx, names):
        n = len(judge_scores)
        J = np.array(judge_scores, dtype=float)
        J_pct = J / J.sum()
        
        if eliminated_idx is None:
            return np.ones(n) / n, np.ones(n) * 0.1, True
        
        def objective(v):
            v_norm = v / v.sum()
            variance_penalty = -0.1 * np.var(v_norm)  
            extreme_penalty = 0.5 * np.sum(np.maximum(0, v_norm - 0.5)**2)  
            return variance_penalty + extreme_penalty
        
        def elimination_constraint(v):
            v_norm = v / v.sum()
            combined = J_pct + v_norm
            elim_score = combined[eliminated_idx]
            #被淘汰者分数必须低于所有其他人
            gaps = []
            for i in range(n):
                if i != eliminated_idx:
                    gaps.append(combined[i] - elim_score)
            return min(gaps) - 0.001
        
        #初始值：基于淘汰约束的启发式
        v0 = np.ones(n) / n
        #被淘汰者给低票
        v0[eliminated_idx] = 0.5 / n
        v0 = v0 / v0.sum()
        
        constraints = [
            {'type': 'eq', 'fun': lambda v: v.sum() - 1.0},
            {'type': 'ineq', 'fun': elimination_constraint}
        ]
        bounds = [(0.01, 0.99) for _ in range(n)]
        
        result = minimize(objective, v0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 2000, 'ftol': 1e-8})
        
        if result.success:
            v_est = result.x / result.x.sum()
            combined = J_pct + v_est
            is_valid = np.argmin(combined) == eliminated_idx
            
            #计算每位选手的不确定性（基于其距离约束边界的距离）
            uncertainties = self._compute_uncertainties(J_pct, v_est, eliminated_idx, n)
            
            return v_est, uncertainties, is_valid
        else:
            #使用备选方案：强制满足约束
            return self._fallback_estimate(J_pct, eliminated_idx, n)
    
    def estimate_week_rank(self, judge_scores, eliminated_idx, names):
        n = len(judge_scores)
        J = np.array(judge_scores, dtype=float)
        
        #评委排名（分数高=排名好=数值小）
        J_rank = rankdata(-J, method='min')
        
        if eliminated_idx is None:
            v_est = J / J.sum()
            return v_est, np.ones(n) * 0.1, True
        
        #被淘汰者需要的条件：合并排名最高（最差）
        elim_J_rank = J_rank[eliminated_idx]
        
        #贪心搜索可行的粉丝排名分配
        best_v = None
        best_score = float('inf')
        
        for trial in range(2000):
            #生成随机粉丝投票
            v_trial = np.random.dirichlet(np.ones(n) * 2)
            V_rank = rankdata(-v_trial, method='min')
            
            combined_rank = J_rank + V_rank
            
            #检查被淘汰者是否合并排名最高
            if np.argmax(combined_rank) == eliminated_idx:
                #评分：希望其他选手的投票有合理性
                score = np.var(v_trial)
                if score < best_score:
                    best_score = score
                    best_v = v_trial
        
        if best_v is not None:
            uncertainties = np.ones(n) * 0.08
            return best_v, uncertainties, True
        else:
            #强制构造解
            v_est = np.ones(n) / n
            v_est[eliminated_idx] = 0.01
            v_est = v_est / v_est.sum()
            return v_est, np.ones(n) * 0.15, False
    
    def _compute_uncertainties(self, J_pct, v_est, eliminated_idx, n):
        uncertainties = np.zeros(n)
        combined = J_pct + v_est
        elim_combined = combined[eliminated_idx]
        
        for i in range(n):
            if i == eliminated_idx:
                uncertainties[i] = 0.03
            else:
                margin = combined[i] - elim_combined
                uncertainties[i] = max(0.02, min(0.15, margin * 0.5))
        
        return uncertainties
    
    def _fallback_estimate(self, J_pct, eliminated_idx, n):
        v_est = np.ones(n) / n
        v_est[eliminated_idx] = 0.01
        v_est = v_est / v_est.sum()
        return v_est, np.ones(n) * 0.1, True
    
    def estimate_season(self, season_data):
        method = season_data['method']
        results = []
        
        for week, week_data in season_data['weeks'].items():
            names = week_data['names']
            judge_scores = week_data['judge_scores']
            eliminated = week_data['eliminated']
            
            eliminated_idx = None
            if eliminated:
                for i, name in enumerate(names):
                    if name == eliminated:
                        eliminated_idx = i
                        break

            if method == 'percentage':
                v_est, uncertainties, is_valid = self.estimate_week_percentage(
                    judge_scores, eliminated_idx, names)
            else:
                v_est, uncertainties, is_valid = self.estimate_week_rank(
                    judge_scores, eliminated_idx, names)
            
            J_pct = judge_scores / judge_scores.sum()
            
            for i, name in enumerate(names):
                results.append({
                    'season': season_data['season'],
                    'week': week,
                    'method': method,
                    'celebrity_name': name,
                    'judge_score': judge_scores[i],
                    'judge_pct': J_pct[i],
                    'fan_vote_est': v_est[i],
                    'fan_vote_std': uncertainties[i],
                    'is_eliminated': name == eliminated,
                    'is_valid': is_valid,
                    'final_placement': season_data['final_placements'].get(name, np.nan)
                })
        
        return results



class UncertaintyQuantifier:
    
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
    
    def compute_uncertainty(self, judge_scores, eliminated_idx, method='percentage'):
        n = len(judge_scores)
        J = np.array(judge_scores, dtype=float)
        J_pct = J / J.sum()
        
        valid_samples = []
        
        for _ in range(self.n_samples * 20):
            #从宽泛的先验采样
            v_sample = np.random.dirichlet(np.ones(n) * 2)
            
            #检查是否满足淘汰约束
            if eliminated_idx is not None:
                if method == 'percentage':
                    combined = J_pct + v_sample
                    if np.argmin(combined) == eliminated_idx:
                        valid_samples.append(v_sample)
                else:
                    J_rank = rankdata(-J, method='min')
                    V_rank = rankdata(-v_sample, method='min')
                    combined_rank = J_rank + V_rank
                    if np.argmax(combined_rank) == eliminated_idx:
                        valid_samples.append(v_sample)
            else:
                valid_samples.append(v_sample)
            
            if len(valid_samples) >= self.n_samples:
                break
        
        if len(valid_samples) < 10:
            return None, None, None, 0.0
        
        samples = np.array(valid_samples)
        
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)
        ci_lower = np.percentile(samples, 2.5, axis=0)
        ci_upper = np.percentile(samples, 97.5, axis=0)
        
        acceptance_rate = len(valid_samples) / (self.n_samples * 20)
        
        return mean, std, (ci_lower, ci_upper), acceptance_rate



def run_complete_analysis(df):
    
    print("=" * 70)
    print("DWTS 粉丝投票估算模型 - 完整分析")
    print("=" * 70)
    
    estimator = ImprovedFanVoteEstimator()
    uncertainty_quantifier = UncertaintyQuantifier(n_samples=500)
    
    all_results = []
    validation_stats = []
    
    for season in range(1, 35):
        season_data = get_season_data(df, season)
        
        if len(season_data['weeks']) == 0:
            continue
        
        #估算该季粉丝投票
        season_results = estimator.estimate_season(season_data)
        all_results.extend(season_results)
        
        #验证
        valid_count = sum(1 for r in season_results if r['is_valid'] and r['is_eliminated'])
        total_elim = sum(1 for r in season_results if r['is_eliminated'])
        
        validation_stats.append({
            'season': season,
            'method': season_data['method'],
            'valid': valid_count,
            'total': total_elim,
            'rate': valid_count / total_elim if total_elim > 0 else 1.0
        })
    
    results_df = pd.DataFrame(all_results)
    validation_df = pd.DataFrame(validation_stats)
    
    return results_df, validation_df


def analyze_controversy_cases(results_df, df):
    
    cases = [
        ('Jerry Rice', 2),
        ('Billy Ray Cyrus', 4),
        ('Bristol Palin', 11),
        ('Bobby Bones', 27)
    ]
    
    print("\n" + "=" * 70)
    print("争议案例深度分析")
    print("=" * 70)
    
    for name, season in cases:
        case_data = results_df[(results_df['celebrity_name'] == name) & 
                               (results_df['season'] == season)]
        
        if len(case_data) == 0:
            print(f"\n{name} (S{season}): 数据未找到")
            continue
        
        print(f"\n{'='*50}")
        print(f"{name} (Season {season})")
        print(f"{'='*50}")
        
        # 获取原始数据中的信息
        orig = df[(df['celebrity_name'] == name) & (df['season'] == season)].iloc[0]
        print(f"最终名次: {orig['placement']}")
        print(f"结果: {orig['results']}")
        
        # 每周分析
        print(f"\n{'Week':<6} {'Judge%':<10} {'FanVote%':<12} {'Diff':<10} {'J_Rank':<8} {'V_Rank':<8}")
        print("-" * 54)
        
        total_j_rank = 0
        total_v_rank = 0
        weeks_competed = 0
        
        for _, row in case_data.iterrows():
            week = row['week']
            j_pct = row['judge_pct']
            v_pct = row['fan_vote_est']
            diff = v_pct - j_pct
            
            #计算该周排名
            week_all = results_df[(results_df['season'] == season) & 
                                   (results_df['week'] == week)]
            n = len(week_all)
            
            j_rank = (week_all['judge_pct'] > j_pct).sum() + 1
            v_rank = (week_all['fan_vote_est'] > v_pct).sum() + 1
            
            total_j_rank += j_rank
            total_v_rank += v_rank
            weeks_competed += 1
            
            print(f"{week:<6} {j_pct*100:>8.1f}%  {v_pct*100:>10.1f}%  {diff*100:>+8.1f}%  {j_rank}/{n:<5} {v_rank}/{n}")
        
        print("-" * 54)
        avg_j_rank = total_j_rank / weeks_competed
        avg_v_rank = total_v_rank / weeks_competed
        print(f"平均评委排名: {avg_j_rank:.1f}")
        print(f"平均粉丝排名: {avg_v_rank:.1f}")
        print(f"粉丝排名优势: {avg_j_rank - avg_v_rank:+.1f} (正值表示粉丝投票更好)")
        
        #总体统计
        avg_j_pct = case_data['judge_pct'].mean()
        avg_v_pct = case_data['fan_vote_est'].mean()
        print(f"\n平均评委得分占比: {avg_j_pct*100:.1f}%")
        print(f"平均粉丝投票占比: {avg_v_pct*100:.1f}%")
        print(f"粉丝投票优势: {(avg_v_pct - avg_j_pct)*100:+.1f}%")


def compute_detailed_uncertainty(results_df, df):
    
    print("\n" + "=" * 70)
    print("不确定性分析")
    print("=" * 70)
    
    #按周统计
    print("\n按比赛周次的不确定性:")
    print(f"{'Week':<6} {'Avg Std':<12} {'Min Std':<12} {'Max Std':<12} {'N':<6}")
    print("-" * 48)
    
    for week in sorted(results_df['week'].unique()):
        week_data = results_df[results_df['week'] == week]
        avg_std = week_data['fan_vote_std'].mean()
        min_std = week_data['fan_vote_std'].min()
        max_std = week_data['fan_vote_std'].max()
        n = len(week_data)
        print(f"{week:<6} {avg_std:>10.4f}  {min_std:>10.4f}  {max_std:>10.4f}  {n:<6}")
    
    #按选手类型统计
    print("\n按选手是否被淘汰的不确定性:")
    for is_elim, label in [(True, '被淘汰者'), (False, '非淘汰者')]:
        subset = results_df[results_df['is_eliminated'] == is_elim]
        print(f"  {label}: 平均std = {subset['fan_vote_std'].mean():.4f}")
    
    # 按方法统计
    print("\n按评分方法的不确定性:")
    for method in ['percentage', 'rank']:
        subset = results_df[results_df['method'] == method]
        print(f"  {method}: 平均std = {subset['fan_vote_std'].mean():.4f}, 一致率 = {subset['is_valid'].mean()*100:.1f}%")
    
    return results_df.groupby('week')['fan_vote_std'].agg(['mean', 'std', 'min', 'max'])



if __name__ == "__main__":
    #加载数据
    print("加载数据...")
    df = load_data('2026_MCM_Problem_C_Data.csv')
    
    #运行完整分析
    results_df, validation_df = run_complete_analysis(df)
    
    #输出验证结果
    print("\n" + "=" * 70)
    print("淘汰一致性验证")
    print("=" * 70)
    
    overall_valid = validation_df['valid'].sum()
    overall_total = validation_df['total'].sum()
    print(f"\n整体一致率: {overall_valid}/{overall_total} = {overall_valid/overall_total*100:.1f}%")
    
    print(f"\n{'Season':<8} {'Method':<12} {'Valid':<8} {'Total':<8} {'Rate':<8}")
    print("-" * 44)
    for _, row in validation_df.iterrows():
        print(f"S{int(row['season']):<7} {row['method']:<12} {int(row['valid']):<8} {int(row['total']):<8} {row['rate']*100:>5.1f}%")
    
    #分析争议案例
    analyze_controversy_cases(results_df, df)
    
    #不确定性分析
    uncertainty_stats = compute_detailed_uncertainty(results_df, df)
    
    #保存结果
    results_df.to_csv('result/fan_vote_estimates_improved.csv', index=False)
    validation_df.to_csv('result/validation_stats.csv', index=False)
    
    print("\n" + "=" * 70)
    print("结果已保存:")
    print("  - fan_vote_estimates_improved.csv (详细估算结果)")
    print("  - validation_stats.csv (验证统计)")
    print("=" * 70)