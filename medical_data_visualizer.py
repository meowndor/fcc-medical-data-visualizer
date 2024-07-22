import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = None

# 3

def BMI(row):
    bmi = row.weight/((row.height/100)**2)

    return 1 if bmi > 25 else 0

df['overweight'] = df.apply(BMI, axis=1)

# 4
def draw_cat_plot():
    # 5
    # df_cat_0 = pd.DataFrame({
    #     'cardio': [0, 0, 1, 1],
    #     'variable': ['cholesterol', 'gluc', 'cholesterol', 'gluc'],
    #     'value': [1, 2, 1, 2],
    #     'total': [10, 20, 30, 40]
    # })
    

    df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
    df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

    # 6
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'overweight'], value_name='value')
    

    # 7
    df_cat = df_cat.groupby(['variable', 'value', 'cardio'])['value'].count().reset_index(name='total')


    # 8
    df_cat['total'] = df_cat['total']/df_cat.groupby('cardio')['total'].transform('sum')*100

    # first plot
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar', height=6, aspect=1).figure


    # 9
    fig.savefig('catplot.png')
    return fig
    # plt.show()


# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # 12
    corr = df_heat.corr(method="pearson")
    # corr = corr.applymap(lambda x: 0.0 if x == -0.0 else x)

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    fig, ax = plt.subplots(figsize=(11, 9))

    # 15
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm', vmax=.3, center=0,
                square=True, linewidths=1, cbar_kws={"shrink": .5}, ax=ax)

    # 16
    fig.savefig('heatmap.png')
    return fig
