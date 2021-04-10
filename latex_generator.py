def generate_latex(
        model_name,
        file_path,
        data_list: list
):
    table = "\\begin{{table}}[]\n\\centering\n\\begin{{tabular}}{{|l|l|l|l|l|l|l|}}\n\\hline\\hline\n" \
                   "Accuracy & Precision & Recall & ROC & AUC \\\\ \\hline\n {} \n\\end{{tabular}}" \
                   "\n\\end{{table}}\n"
    table_column = "{} & {} & {} & {} & {} \\\\ \\hline\n"
    columns = ""
    for data in data_list:
        columns += table_column.format(
            data['acc'],
            data['pre'],
            data['rec'],
            data['roc'],
            data['auc']
        )
    with open(file_path, 'w+') as f:
        f.write(table.format(columns))
