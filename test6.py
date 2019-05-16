import statsmodels.formula.api as smf
import pandas as pd


def forward_selected(data, response):
    """前向逐步回归算法，源代码来自https://planspace.org/20150423-forward_selection_with_statsmodels/
    使用Adjusted R-squared来评判新加的参数是否提高回归中的统计显著性
    Linear model designed by forward selection.
    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response
    response: string, name of response column in data
    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()

    return model


def main():
    '''
    首先从网上读取普林斯顿大学52位员工的工资信息，工资文件一共有6列，各列意义解释如下：
    sx = Sex, coded 1 for female and 0 for male
    rk = Rank, coded
        1 for assistant professor,
        2 for associate professor, and
        3 for full professor
    yr = Number of years in current rank
    dg = Highest degree, coded 1 if doctorate, 0 if masters
    yd = Number of years since highest degree was earned
    sl = Academic year salary, in dollars.
    '''
    url = "http://data.princeton.edu/wws509/datasets/salary.dat"
    data = pd.read_csv(url, sep='\\s+')

    # 将sl（年收入）设为目标变量
    model = forward_selected(data, 'sl')

    # 打印出最后的回归模型
    print(model.model.formula)
    # sl ~ rk + yr + 1
    print(model.params)
    # Intercept          16203.268154
    # rk[T.associate]     4262.284707
    # rk[T.full]          9454.523248
    # yr                   375.695643
    # dtype: float64
    # 0.835190760538

    print(model.rsquared_adj)
    # 0.835190760538


if __name__ == '__main__':
    main()
