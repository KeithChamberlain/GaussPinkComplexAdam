"""Analysis of Initializer and Optimizer Combinations."""
import sys
import json
import numpy as np
import pandas as pd
import openpyxl
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import statsmodels.formula.api as smf


if __name__ == "__main__":
    workbook = str(sys.argv[1])
    output_workbook = str(sys.argv[2])

    formulas = {
        "loss": "val_loss ~ gauss_vs_pink + adam_vs_complex + interaction",
        "acc": "val_acc ~ gauss_vs_pink + adam_vs_complex + interaction",
        "secs": "train ~ gauss_vs_pink + adam_vs_complex + interaction"
    }


    codes = pd.DataFrame({
        "intercept": np.repeat(1, 4),
        "gauss_vs_pink": np.array([.5, .5, -.5, -.5]),
        "adam_vs_complex": np.array([.5, -.5, .5, -.5]),
        "interaction": np.array([.25, -.25, -.25, .25]),
    })


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
    workbook_as_df = pd.concat(df_list)
    
    # Make Grouping Column
    workbook_as_df['group'] = workbook_as_df['init'] + ", " + workbook_as_df['opt'].str.replace("torch_", "").replace("complex_adam", "complex")
    gb = workbook_as_df.groupby("group").agg("count")
    print(gb)
    print(workbook_as_df.head(10))
    
    
    # Set Contrasts
    #       gauss_vs_pink. adam-vs_complex.           int
    # gauss, adam      .5              .5            .25
    # gauss, complex   .5             -.5           -.25
    # pink, adam      -.5              .5           -.25
    # pink, complex   -.5             -.5            .25
    codes = pd.DataFrame({
        "means": workbook_as_df.loc[:, "group"].unique(),
        "intercept": np.repeat(1, 4),
        "gauss_vs_pink": np.array([-.5, -.5, +.5, +.5]),
        "adam_vs_complex": np.array([-.5, +.5, -.5, +.5]),
        "interaction": np.array([-.25, +.25, +.25, -.25]),
    })
    print(codes) # Check the order
    
    workbook_as_df['gauss_vs_pink'] = (
        -.5*(workbook_as_df['group'] == "gauss, adam").astype('int') -
        .5*(workbook_as_df['group']== "gauss, complex").astype('int') +
        .5*(workbook_as_df['group'] == "pink, adam").astype('int') +
        .5*(workbook_as_df['group']== "pink, complex").astype('int')
    )
    workbook_as_df['adam_vs_complex'] = (
        -.5*(workbook_as_df['group'] == "gauss, adam").astype('int') +
        .5*(workbook_as_df['group']=="gauss, complex").astype('int') -
        .5*(workbook_as_df['group'] == "pink, adam").astype('int') +
        .5*(workbook_as_df['group']=="pink, complex").astype('int')
    )
    workbook_as_df['interaction'] = (
        -.25*(workbook_as_df['group'] == "gauss, adam").astype('int') +
        .25*(workbook_as_df['group']=="gauss, complex").astype('int') +
        .25*(workbook_as_df['group'] == "pink, adam").astype('int') -
        .25*(workbook_as_df['group']=="pink, complex").astype('int')
    )

    # Original Models
    loss_orig_fit = smf.ols(formulas["loss"], data=workbook_as_df).fit()
    acc_orig_fit = smf.ols(formulas["acc"], data=workbook_as_df).fit()
    secs_orig_fit = smf.ols(formulas["secs"], data=workbook_as_df).fit()

    # Review Original Fits
    loss_sum = loss_orig_fit.summary2().tables[1]
    acc_sum = acc_orig_fit.summary2().tables[1]
    secs_sum = secs_orig_fit.summary2().tables[1]

    loss = codes.iloc[:, 1:].to_numpy().dot(loss_sum["Coef."])
    acc = codes.iloc[:, 1:].to_numpy().dot(acc_sum["Coef."])
    secs = codes.iloc[:, 1:].to_numpy().dot(secs_sum["Coef."])
    loss_ci_l = codes.iloc[:, 1:].to_numpy().dot(loss_sum["[0.025"])
    loss_ci_u = codes.iloc[:, 1:].to_numpy().dot(loss_sum["0.975]"])
    print("loss: sum, means, cil, ciu")
    print(loss_sum)
    print(loss)
    print(loss_ci_l)
    print(loss_ci_u)
    
    print(acc_sum)
    print(secs_sum)

    means = workbook_as_df.groupby("group")[['val_loss', 'val_acc', 'train']].agg(["mean", "std"])
    means['init'] = ["gauss", "gauss", "pink", "pink"]
    means['opt'] = ["adam", "complex", "adam", "complex"]
    init_means = means.groupby("init")[[('val_loss', 'mean'), ('val_loss', 'std'), ('val_acc', 'mean'), ('val_acc', 'std'), ('train', 'mean'), ('train', 'std')]].agg("mean")
    opt_means = means.groupby("opt")[[('val_loss', 'mean'), ('val_loss', 'std'), ('val_acc', 'mean'), ('val_acc', 'std'), ('train', 'mean'), ('train', 'std')]].agg("mean")

    print(means)
    print(means.index)
    print(means.columns)
    print(init_means)
    print(init_means.columns)
    print(init_means.index)

    figx = make_subplots(rows=1, cols=3,
        x_title = "Mean Differences",
        y_title = "Scores",
        specs = [[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]],
        subplot_titles = ["Loss", "Accuracy", "Trainign Time (s)"]
    )


    index = loss_sum.index[1:]
    figx.add_trace(go.Bar(
        name="Loss (Left)",
        x=loss_sum.index[1:],
        y=loss_sum.loc[index, 'Coef.'],
        #mode="markers",
        error_y=dict(
            type="data",
            symmetric=False,
            array=loss_sum.loc[index, '0.975]'] - loss_sum.loc[index, 'Coef.'],
            arrayminus=loss_sum.loc[index,'Coef.'] - loss_sum.loc[index, '[0.025']
        )
    ), secondary_y=False, row=1, col=1)


    index = acc_sum.index[1:]
    figx.add_trace(go.Bar(
        name="Accuracy (Center)",
        x=acc_sum.index[1:],
        y=acc_sum.loc[index, 'Coef.'],
        #mode="markers",
        error_y=dict(
            type="data",
            symmetric=False,
            array=acc_sum.loc[index, '0.975]'] - acc_sum.loc[index, 'Coef.'],
            arrayminus=acc_sum.loc[index,'Coef.'] - acc_sum.loc[index, '[0.025']
        )
    ), secondary_y=False, row=1, col=2)


    index = secs_sum.index[1:]
    figx.add_trace(go.Bar(
        name="Training Time (s) (Right)",
        x=secs_sum.index[1:],
        y=secs_sum.loc[index, 'Coef.'],
        #mode="markers",
        error_y=dict(
            type="data",
            symmetric=False,
            array=secs_sum.loc[index, '0.975]'] - secs_sum.loc[index, 'Coef.'],
            arrayminus=secs_sum.loc[index,'Coef.'] - secs_sum.loc[index, '[0.025']
        )
    ), secondary_y=False, row=1, col=3)

    figx.update_layout(
        title=dict(
            text="Mean differences for transformer neural network metrics for initializer and optimizer types",
            subtitle=dict(text="With 95% confidence interval")))

    fig_orig = make_subplots(rows=2, cols=1,
        x_title = "Group Means",
        y_title = "Scores",
        specs=[[{"secondary_y": True}], [{"secondary_y":False}]],
        subplot_titles = ["Loss and accuracy metrics", "Time metric"]
    )

    fig_orig.add_trace(go.Scatter(
        name="Loss (Left)",
        x=means.index,
        y=means.loc[:, ('val_loss', 'mean')],
        mode="markers",
        error_y=dict(
            type='data',
            symmetric=True,
            array=loss_sum['Std.Err.']
        )
    ), secondary_y=False, row=1, col=1)


    fig_orig.add_trace(go.Scatter(
        name="Accuracy (Right)",
        x=means.index,
        y=means.loc[:, ('val_acc', 'mean')],
        mode="markers",
        error_y=dict(
            type='data',
            symmetric=True,
            array=acc_sum['Std.Err.']
        )
    ), secondary_y = True, row=1, col=1)


    fig_orig.add_trace(go.Scatter(
        name="Training Time (s) (Bottom)",
        x=means.index,
        y=means.loc[:, ('train', 'mean')],
        mode="markers",
        error_y=dict(
            type='data',
            symmetric=True,
            array=secs_sum['Std.Err.']
        )
    ), secondary_y = False, row=2, col=1)

    fig_orig.update_yaxes(title_text = "Loss", row=1, col=1, secondary_y=False)
    fig_orig.update_yaxes(title_text = "Accuracy", row=1, col=1, secondary_y=True)
    fig_orig.update_yaxes(title_text = "Training Time (s)", row=2, col=1, secondary_y=False)
    fig_orig.update_layout(
        title=dict(
            text="Means for transformer neural network metrics for initializer and optimizer types",
            subtitle=dict(text="With standard error bars")))

    with open("combined_dashboard.html", "w") as f:
        f.write("<html><head><title>Dashboard</title></head><body>")
        f.write(figx.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig_orig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("</body></html>")


    loss_sum.to_excel(output_workbook, sheet_name="LOSS_OLS")

    with pd.ExcelWriter(output_workbook, engine="openpyxl", mode="a") as writer:
        acc_sum.to_excel(writer, sheet_name="ACC_OLS")
        secs_sum.to_excel(writer, sheet_name="TRAIN_TIME_OLS")
        means.to_excel(writer, sheet_name="MEANS")
        init_means.to_excel(writer, sheet_name="MEANS_INIT")
        opt_means.to_excel(writer, sheet_name="MEANS_OPT")
