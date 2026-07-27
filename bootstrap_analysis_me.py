"""Bootstrap resampling and analysis utilities."""
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
import openpyxl
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import statsmodels.formula.api as smf

def replicate_samples(df, sample, replicates, seed=None):
    from numpy.random import Generator, PCG64DXSM
    
    rand = PCG64DXSM(seed) if seed is not None else PCG64DXSM()
    rng = Generator(rand)
    dfs = [
        df.sample(n=sample, replace=True, random_state=rng, ignore_index=False)
        for _ in np.arange(replicates)
    ]
    return dfs


def many_ols_models(dfs, formula):
    """
    Fits ols replicates to datat after resampling. Returns a list of ols objects
    for each sample.
    """
    return [
        smf.ols(
            formula=formula,
            data=df
        )
        for df in dfs
    ]


def many_ols_fits(models):
    """
    Fits ols replicates times after resampling. Returns a list of ols objects
    """
    return [model.fit() for model in models]


def summarize_many_ols_coeficients(fit_list, statistic):
    """
    Extract and summarize coeficients from multiple models based on statistic.
    Returns a Series containing the summarized values.
    """
    coefs = pd.concat([fit.summary2().tables[1].reset_index() for fit in fit_list]).groupby("index").agg(statistic)
    lines = coefs.size
    coefs['df_model'] = np.mean([fit.df_model for fit in fit_list])
    coefs['df_resid'] = np.mean([fit.df_resid for fit in fit_list])
    coefs['n_obs'] = np.mean([fit.nobs for fit in fit_list])
    return coefs



def complete_many_ols_coeficients_table(df, confidence=0.95): #, mean_betas_column, sem_boot_column):
    df = df.reset_index()
    results  = pd.DataFrame({
        "index":df.loc[:,('index', '')],
        "betas":df.loc[:, ('Coef.', 'mean')],
        "SE_boot":df.loc[:, ('Coef.', 'std')],
        "t":df.loc[:, ('Coef.', 'mean')] / df.loc[:, ('Coef.', 'std')],
    })
    results["p"] = stats.t.sf(np.abs(results.loc[:,'t']), df.loc[:, 'df_model']) * 2
    results[f"[{np.round((1-confidence)/2, 3)}"] = df.loc[:, (f'[{np.round((1-confidence)/2, 3)}', 'mean')]
    results[f"{np.round(1 - (1 - confidence)/2, 3)}]"] = df.loc[:, (f'{np.round(1 - (1 - confidence)/2, 3)}]', 'mean')]
    return results


def resample_bias(boot_summary, fits, statistic):
    return statistic([
        boot_summary.loc[:, "betas"] - fit.summary2().tables[1].reset_index().loc[:, "Coef."] for fit in fits],
        axis=0
    )


def boot_bias(boot_summary, fit):
    return boot_summary.loc[:, "betas"] - fit.summary2().tables[1].reset_index().loc[:, "Coef."]

def summarize_many_anovas(fit_list, statistic, type="III"):
    """
    Extract and partition SS from multiple fits. Returns a DataFrame containing 
    the table
    """
    import statsmodels.api as sm
    return pd.concat([sm.stats.anova_lm(fit, typ=type).reset_index() for fit in fit_list]).groupby(["index"]).agg(statistic)


def smooth_many_ols_predictions(fit_list, statistic, axis=1):
    """
    Extract and summarize resampled model predictions based on statistic.
    Returns a DataFrame with the smoothed predictions fit with the smoothed
    results.
    """
    # Could: repeated ols to predictions
    # Or: repeated samples, averaged single ols to predictions
    return statistic(np.array([fit.predict() for fit in fit_list]), axis=axis)


def flat_stat(dfs, columns, dvs, statistics, axis=0):
    """
    Generate a statistic from repeated samples.
    """
    value_dict = dict()
    results_list = list()
    for df in dfs:
        for column in columns:
            value_dict[(column)] = df.groupby(column)[dvs].agg(statistics, axis=axis)
        results_list.append(pd.concat(value_dict))
    data = pd.concat(results_list)
    return data.groupby(data.index)[data.columns].agg(statistics, axis=0)


def lower_q(value, axis=0):
    return np.quantile(value, 0.025, axis)


def upper_q(value, axis=0):
    return np.quantile(value, 0.975, axis)


def lower_q2(value, axis=1):
    return np.quantile(value, 0.025, axis)


def upper_q2(value, axis=1):
    return np.quantile(value, 0.975, axis)


def group_sampling_distribution(dfs, group_by, values, statistic):
    samp_dist = list()
    for df in dfs:
        samp_dist.append(pd.concat(df.groupby(group)[values].agg(statistic, axis=0) for group in group_by))
    return pd.concat(samp_dist)


if __name__ == "__main__":
    boot_rows = int(sys.argv[1])
    boot_cols = int(sys.argv[2])
    workbook = str(sys.argv[3])
    output_workbook = str(sys.argv[4])

    formulas = {
        "loss": "val_loss ~ gauss_vs_pink + adam_vs_complex + interaction",
        "acc": "val_acc ~ gauss_vs_pink + adam_vs_complex + interaction",
        "secs": "secs ~ gauss_vs_pink + adam_vs_complex + interaction"
    }
    # Read Workbook
    wkbook = openpyxl.load_workbook(workbook, data_only=True)
    df_list = list()
    for sheet in wkbook.sheetnames:
        print(sheet)
        if sheet == "Sheet1":
            pass
        else:
            rows = list(wkbook[sheet].iter_rows(values_only=True))
            header = rows[0]
            data = rows[1:]
            df = pd.DataFrame(data, columns=header)
            df['sheet'] = sheet
            df_list.append(df)
    wkbook.close()
    # Contrasts
    #       gauss_vs_pink. adam-vs_complex. interaction
    # gauss     .5              0             .5
    # pink      -.5             0             .5
    # adam      0               .5           -.5
    # compl     0              -.5           -.5
    workbook_as_df = pd.concat(df_list).reset_index()
    workbook_as_df['gauss_vs_pink'] = (
        0.5*(workbook_as_df['init'] == "gauss").astype('int') -
        0.5*(workbook_as_df['init']=="pink").astype('int')
    )
    workbook_as_df['adam_vs_complex'] = (
        0.5*(workbook_as_df['opt'] == 'torch_adam').astype('int') -
        0.5*(workbook_as_df['opt']=="complex_adam").astype('int')
    )
    workbook_as_df['interaction'] = (
        0.5*(workbook_as_df['init'] == "gauss").astype('int') -
        0.5*(workbook_as_df['init']=="pink").astype('int') -
        0.5*(workbook_as_df['opt'] == 'torch_adam').astype('int') -
        0.5*(workbook_as_df['opt']=="complex_adam").astype('int')
    )

    print("The number of trials for the transformer NN comparison is: " + str(workbook_as_df.shape[0]))

    # Original Models
    loss_orig_fit = smf.ols(formulas["loss"], data=workbook_as_df).fit()
    acc_orig_fit = smf.ols(formulas["acc"], data=workbook_as_df).fit()
    secs_orig_fit = smf.ols(formulas["secs"], data=workbook_as_df).fit()


    # Replicates
    dfs = replicate_samples(workbook_as_df, sample=boot_rows, replicates=boot_cols)
    print(dfs[0].head(5))
    print()
    # DV Models
    loss_ols_models = many_ols_models(dfs=dfs, formula=formulas['loss'])
    acc_ols_models = many_ols_models(dfs=dfs, formula=formulas['acc'])
    secs_ols_models = many_ols_models(dfs=dfs, formula=formulas['secs'])
    # Fits
    loss_ols_fits = many_ols_fits(loss_ols_models)
    acc_ols_fits = many_ols_fits(acc_ols_models) 
    secs_ols_fits = many_ols_fits(secs_ols_models)
    # Summarize Replicates
    loss_summarized_table = summarize_many_ols_coeficients(fit_list=loss_ols_fits, statistic=[np.mean, np.std, lower_q, upper_q])
    loss_calculated_coef_summary = complete_many_ols_coeficients_table(loss_summarized_table)
    loss_summarized_anova = summarize_many_anovas(fit_list=loss_ols_fits, statistic=np.mean, type="III")
    loss_predicted = smooth_many_ols_predictions(fit_list=loss_ols_fits, statistic=np.mean, axis=1)
    acc_summarized_table = summarize_many_ols_coeficients(fit_list=acc_ols_fits, statistic=[np.mean, np.std])
    acc_calculated_coef_summary = complete_many_ols_coeficients_table(acc_summarized_table)#, mean_betas_column, sem_boot_column)
    acc_summarized_anova = summarize_many_anovas(fit_list=acc_ols_fits, statistic=np.mean, type="III")
    acc_predicted = smooth_many_ols_predictions(fit_list=acc_ols_fits, statistic=np.mean, axis=1)
    secs_summarized_table = summarize_many_ols_coeficients(fit_list=secs_ols_fits, statistic=[np.mean, np.std])
    secs_calculated_coef_summary = complete_many_ols_coeficients_table(secs_summarized_table)#, mean_betas_column, sem_boot_column)
    secs_summarized_anova = summarize_many_anovas(fit_list=secs_ols_fits, statistic=np.mean, type="III")
    secs_predicted = smooth_many_ols_predictions(fit_list=secs_ols_fits, statistic=np.mean, axis=1)

    samp_dist_means = group_sampling_distribution(dfs, ['init', 'opt'], ['val_loss', 'val_acc', 'secs'], np.mean).reset_index()
    print(samp_dist_means)
    
    means = samp_dist_means.groupby("index").agg([np.mean, lower_q, upper_q]).reset_index()
    
    means.loc[:, 'loss_upper'] = means.loc[:, ("val_loss", "upper_q")] - means.loc[:, ("val_loss", "mean")]
    means.loc[:, 'loss_lower'] = means.loc[:, ("val_loss", "mean")] - means.loc[:, ("val_loss", "lower_q")]
    means.loc[:, 'acc_upper'] = means.loc[:, ("val_acc", "upper_q")] - means.loc[:, ("val_acc", "mean")]
    means.loc[:, 'acc_lower'] = means.loc[:, ("val_acc", "mean")] - means.loc[:, ("val_acc", "lower_q")]
    means.loc[:, 'secs_upper'] = means.loc[:, ("secs", "upper_q")] - means.loc[:, ("secs", "mean")]
    means.loc[:, 'secs_lower'] = means.loc[:, ("secs", "mean")] - means.loc[:, ("secs", "lower_q")]
    print(means)
    print(means.columns)

    # add resample/boot bias
    loss_calculated_coef_summary["boot_bias"] = boot_bias(loss_calculated_coef_summary, loss_orig_fit)
    acc_calculated_coef_summary["boot_bias"] = boot_bias(acc_calculated_coef_summary, acc_orig_fit)
    secs_calculated_coef_summary["boot_bias"] = boot_bias(secs_calculated_coef_summary, secs_orig_fit)
    loss_calculated_coef_summary["resample_bias"] = resample_bias(loss_calculated_coef_summary, loss_ols_fits, statistic=np.mean)
    acc_calculated_coef_summary["resample_bias"] = resample_bias(acc_calculated_coef_summary, acc_ols_fits, statistic=np.mean)
    secs_calculated_coef_summary["resample_bias"] = resample_bias(secs_calculated_coef_summary, secs_ols_fits, statistic=np.mean)

    loss_calculated_coef_summary.to_excel(output_workbook, sheet_name="LOSS_OLS")

    with pd.ExcelWriter(output_workbook, engine="openpyxl", mode="a") as writer:
        # loss_summarized_anova.to_excel(writer, sheet_name="LOSS_ANOVA")
        acc_calculated_coef_summary.to_excel(writer, sheet_name="ACC_OLS")
        secs_calculated_coef_summary.to_excel(writer, sheet_name="SEC_OLS")
        # secs_summarized_table.to_excel(writer, sheet_name="SECS_OLS")
        # secs_summarized_anova.to_excel(writer, sheet_name="SECS_ANOVA")
        # stats_results.to_excel(writer, sheet_name="STATS")

    loss_summarized_table = loss_summarized_table.reset_index()

    fig = make_subplots(rows=2, cols=3,
        x_title = "Group Mean Differences",
        y_title = "Scores",
        subplot_titles = [
            "Init loss", "Init accuracy",
            "Init time", "Optimizer loss",
            "Optimizer accuracy", "Optimizer time"
        ]
    )
    print(means.loc[:, "index"])
    index_gp = [True if grp in ['gauss', 'pink']  else False for grp in  means.loc[:, "index"]]
    fig.add_trace(go.Scatter(
        name="Initialization Strategy Loss",
        x=means.loc[index_gp, ('index', '')],
        y=means.loc[index_gp, ("val_loss", "mean")],
        error_y=dict(
            type='data',
            symmetric=False,
            array=means.loc[index_gp, 'loss_upper'],
            arrayminus=means.loc[index_gp, 'loss_lower']
        )
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        name="Initialization Strategy Accuracy",
        x=means.loc[index_gp, ('index', '')],
        y=means.loc[index_gp, ('val_acc', 'mean')],
        error_y=dict(
            type='data',
            symmetric=False,
            array=means.loc[index_gp, 'acc_upper'],
            arrayminus=means.loc[index_gp, 'acc_lower']
        )
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        name="Initialization Strategy Time(s)",
        x=means.loc[index_gp, ('index', '')],
        y=means.loc[index_gp, ('secs', 'mean')],
        error_y=dict(
            type='data',
            symmetric=False,
            array=means.loc[index_gp, 'secs_upper'],
            arrayminus=means.loc[index_gp, 'secs_lower']
        )
    ), row=1, col=3)

    index_ac = [True if grp in ['torch_adam', 'complex_adam']  else False for grp in  means.loc[:, "index"]]
    fig.add_trace(go.Scatter(
        name="Optimization Strategy Loss",
        x=means.loc[index_ac, ('index', '')],
        y=means.loc[index_ac, ('val_loss', 'mean')],
        error_y=dict(
            type='data',
            symmetric=False,
            array=means.loc[index_ac, 'loss_upper'],
            arrayminus=means.loc[index_ac, 'loss_lower']
        )
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        name="Optimization Strategy Accuracy",
        x=means.loc[index_ac, ('index', '')],
        y=means.loc[index_ac, ('val_acc', 'mean')],
        error_y=dict(
            type='data',
            symmetric=False,
            array=means.loc[index_ac, 'acc_upper'],
            arrayminus=means.loc[index_ac, 'acc_lower']
        )
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        name="Optimization Strategy Time",
        x=means.loc[index_ac, ('index', '')],
        y=means.loc[index_ac, ('secs', 'mean')],
        error_y=dict(
            type='data',
            symmetric=False,
            array=means.loc[index_ac, 'secs_upper'],
            arrayminus=means.loc[index_ac, 'secs_lower']
        )
    ), row=2, col=3)

    fig.update_layout(
        title=dict(
            text="Gaussian initialization shows advantage to pink initialization across loss, accuracy and time;\ncomplex adam show an advantage to adam except for time.",
            subtitle=dict(text="Means with bootstrapped approximate 95% CIs based on percenitles")
    ))
    fig.write_html("adam_complex_gauss_pink.html")
    fig.show()

    ##########----------Histograms

    fig_hists = make_subplots(rows=3, cols=4,
        x_title = "Group Mean Histograms",
        y_title = "Frequency",
        subplot_titles = [
            "Gaussian Loss", "Pink Loss", "Adam Loss", "Complex Loss",
            "Gaussian Accuracy", "Pink Accuracy", "Adam Accuracy", "Complex Accuracy",
            "Gaussian Time(s)", "Pink Time(s)", "Adam Time(s)", "Complex Time(s)"
        ]
    )
    for i, metric in enumerate(["val_loss", "val_acc", "secs"]):
        for j, group in enumerate(['gauss', 'pink', 'torch_adam', 'complex_adam']):
            data = samp_dist_means.loc[samp_dist_means.loc[:, 'index'] == group, metric]
            fig_hists.add_trace(go.Histogram(
                #name="Gaussian Loss",
                x=data,
                nbinsx=50,
            ), row=i+1, col=j+1)
    fig_hists.update_layout(
        title=dict(text="Histograms of data are used without transformation",
        subtitle=dict(text="data do not indicate long tails, are far from bounds")
    ))
    fig_hists.show()
    fig_hists.write_html("adam_complex_gauss_pink_hists.html")
