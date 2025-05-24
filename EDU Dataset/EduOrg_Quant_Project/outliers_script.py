import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


# In[2]:


npo = pd.read_csv("EDNPOs.csv")
npo = npo.set_index(["nccs_level_1","name","ein2"])
npo = npo.sort_index(ascending=False)
npo



org = np.array(["Scholarship","Single Support Fundraising","Fundraising","Education Services","Student Services","Advocacy","Research"])
service_org = npo[npo.Main_Category.isin(org)]
service_org["city"].value_counts()
top_10_cities = service_org["city"].value_counts().head(50)
service_org = service_org[service_org["city"].isin(top_10_cities.index)]
service_org = service_org.reset_index()
service_org["city"].value_counts()


# In[41]:




# In[73]:


service_org.Main_Category.value_counts()
new_york = service_org[service_org["city"] == "NEW YORK"]
atlanta = service_org[service_org["city"] == "ATLANTA"]
chicago = service_org[service_org["city"] == "CHICAGO"]
houston = service_org[service_org["city"] == "HOUSTON"]
dallas = service_org[service_org["city"] == "DALLAS"]
washington = service_org[service_org["city"] == "WASHINGTON"]
losangeles = service_org[service_org["city"] == "LOS ANGELES"]
lasvegas = service_org[service_org["city"] == "LAS VEGAS"]
total = pd.concat([new_york, atlanta, chicago, houston, dallas, washington, losangeles, lasvegas])
new_york


# In[74]:


def get_quant_cols(df):
    return df[['totrev', 'total_income', 'total_assets', 'org_year_count', 'org_fiscal_period', 'org_year_first', 'org_year_last']]


# In[75]:


"""Get new cols through feature engineering"""
def engineer_features(df):
    df["total_income_per_asset"] = df["total_income"] / df["total_assets"]
    df["total_income_per_year"] = df["total_income"] / df["org_year_count"]
    df["total_income/totrev"] = df["total_income"] / df["totrev"]
    df["total_income/total_asstes"] = df["total_income"] / df["total_assets"]
    return df

# Replace positive and negative infinity values with 1
def replace_infinity_with_one(df):
    return df.replace([np.inf, -np.inf], 1)


# In[76]:


quant_newyork = replace_infinity_with_one(engineer_features(get_quant_cols(new_york))).fillna(1)
quant_atlanta = replace_infinity_with_one(engineer_features(get_quant_cols(atlanta))).fillna(1)
quant_chicago = replace_infinity_with_one(engineer_features(get_quant_cols(chicago))).fillna(1)
quant_houston = replace_infinity_with_one(engineer_features(get_quant_cols(houston))).fillna(1)
quant_dallas = replace_infinity_with_one(engineer_features(get_quant_cols(dallas))).fillna(1)
quant_washington = replace_infinity_with_one(engineer_features(get_quant_cols(washington))).fillna(1)
quant_losangeles = replace_infinity_with_one(engineer_features(get_quant_cols(losangeles))).fillna(1)
quant_lasvegas = replace_infinity_with_one(engineer_features(get_quant_cols(lasvegas))).fillna(1)
quant_total = pd.concat([quant_newyork, quant_atlanta, quant_chicago, quant_houston, quant_dallas, quant_washington, quant_losangeles, quant_lasvegas], axis=0)


# In[77]:


import numpy as np
from sklearn.ensemble import IsolationForest
# Separate the non-numeric columns (e.g., organization names)
"""Count nas in research"""
def get_outlier(df,df_total):
    model = IsolationForest(contamination=0.01, n_estimators=100, random_state=42)
    model.fit(df.select_dtypes(include=[np.number]))  # Fit only on numeric columns
    df["outlier"] = model.predict(df.select_dtypes(include=[np.number]))
    outliers = df[df['outlier'] == -1]
    
    # Extract the indices of the outliers
    outlier_indices = outliers.index.tolist()
    print(f"Number of outliers: {outliers.shape[0]}")
    df_total["outlier"] = False
    df_total.loc[outlier_indices, "outlier"] = True
    return outlier_indices



quant_newyork


# In[78]:


new_york_outliers = get_outlier(quant_newyork,new_york)
atlanta_outliers = get_outlier(quant_atlanta,atlanta)
chicago_outliers = get_outlier(quant_chicago,chicago)
houston_outliers = get_outlier(quant_houston,houston)
dallas_outliers = get_outlier(quant_dallas,dallas)
washington_outliers = get_outlier(quant_washington,washington)
losangeles_outliers = get_outlier(quant_losangeles,losangeles)
lasvegas_outliers = get_outlier(quant_lasvegas,lasvegas) 
total_outliers = get_outlier(quant_total,total)


# In[96]:
total_outliers = total[total["outlier"] == True]



# In[ ]:





# In[84]:


import matplotlib.pyplot as plt

def plot_outliers(df_total, x_col, y_col, city_name="City"):
    # Check that the selected columns exist
    if x_col not in df_total.columns or y_col not in df_total.columns:
        print(f"Columns '{x_col}' or '{y_col}' not found in the dataframe for {city_name}.")
        return

    # Plot
    plt.figure(figsize=(10, 6))
    
    # Non-outliers
    plt.scatter(
        df_total.loc[~df_total["outlier"], x_col],
        df_total.loc[~df_total["outlier"], y_col],
        label="Normal",
        alpha=0.6
    )
    
    # Outliers
    plt.scatter(
        df_total.loc[df_total["outlier"], x_col],
        df_total.loc[df_total["outlier"], y_col],
        color="red",
        label="Outliers",
        alpha=0.9
    )

    plt.title(f"{city_name} - {x_col} vs. {y_col} (Outliers in Red)")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()




# In[90]:


plot_outliers(new_york, "org_year_count", "total_income", "New York")
plot_outliers(new_york, "total_income", "total_assets", "New York")
plot_outliers(new_york, "org_year_count", "totrev", "New York")
plot_outliers(new_york, "org_year_count", "bmf_deductibility_code", "New York")



# In[ ]:


plot_outliers(new_york, "org_year_count", "total_income", "New York")
plot_outliers(new_york, "total_income", "total_assets", "New York")
plot_outliers(new_york, "org_year_count", "totrev", "New York")
plot_outliers(new_york, "org_year_count", "bmf_deductibility_code", "New York")



# In[97]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the plot
plt.figure(figsize=(12, 8))

# Plot all non-outliers in gray
sns.scatterplot(
    data=total[total['outlier'] == False],
    x='totrev',
    y='total_income',
    color='gray',
    alpha=0.5,
    label='Non-Outliers'
)

# List of your 8 cities
cities = ["NEW YORK", "CHICAGO", "HOUSTON", "DALLAS", "WASHINGTON", "LOS ANGELES", "LAS VEGAS", "ATLANTA"]
palette = sns.color_palette("tab10", n_colors=len(cities))

# Plot each city's outliers in a unique color
for i, city in enumerate(cities):
    city_outliers = total[(total['outlier'] == True) & (total['city'] == city)]
    sns.scatterplot(
        data=city_outliers,
        x='totrev',
        y='total_income',
        label=f'Outlier - {city}',
        color=palette[i],
        edgecolor='black'
    )

# Customize appearance
plt.title("Total Revenue vs Total Income\nOutliers Highlighted by City")
plt.xlabel("Total Revenue (totrev)")
plt.ylabel("Total Income")
plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.tight_layout()
plt.grid(True)
plt.show()


# In[98]:


import plotly.express as px
import pandas as pd

# Filter outliers and non-outliers
non_outliers = total[total['outlier'] == False]
outliers = total[total['outlier'] == True]

# Base scatter for non-outliers (in gray)
fig = px.scatter(
    non_outliers,
    x='org_year_count',
    y='total_income',
    opacity=0.4,
    color_discrete_sequence=['gray'],
    hover_data=['name', 'city', 'org_year_count', 'total_income'],
    labels={'org_year_count': 'Years in Operation', 'total_income': 'Total Income'},
)

# Overlay outliers grouped by city (each city gets its own color)
fig_outliers = px.scatter(
    outliers,
    x='org_year_count',
    y='total_income',
    color='city',
    hover_data=['name', 'city', 'org_year_count', 'total_income'],
    labels={'org_year_count': 'Years in Operation', 'total_income': 'Total Income'},
)

# Add the outliers as a new trace group into the base figure
for trace in fig_outliers.data:
    fig.add_trace(trace)

# Customize layout
fig.update_layout(
    title="Interactive Scatterplot: Years in Operation vs Total Income (Outliers Highlighted by City)",
    legend_title="City (Outliers)",
    xaxis_title="Years in Operation (org_year_count)",
    yaxis_title="Total Income",
    template="plotly_white",
    height=700
)

# Show plot
fig.show()


# In[103]:


total_outliers.to_csv("total_outliers.csv", index=False)


# In[105]:


import plotly.express as px
import pandas as pd

# Separate outliers and non-outliers
non_outliers = total[total['outlier'] == False]
outliers = total[total['outlier'] == True]

# Base plot: gray points for non-outliers
fig = px.scatter(
    non_outliers,
    x='totrev',
    y='total_income',
    color_discrete_sequence=['gray'],
    opacity=0.4,
    hover_data=['name', 'city', 'totrev', 'total_income'],
    labels={'totrev': 'Total Revenue', 'total_income': 'Total Income'},
)

# Overlay: outliers colored by city
fig_outliers = px.scatter(
    outliers,
    x='totrev',
    y='total_income',
    color='city',
    hover_data=['name', 'city', 'totrev', 'total_income'],
    labels={'totrev': 'Total Revenue', 'total_income': 'Total Income'},
)

# Add outlier traces to the base plot
for trace in fig_outliers.data:
    fig.add_trace(trace)

# Final plot styling
fig.update_layout(
    title="Interactive Scatterplot: Total Revenue vs Total Income (Outliers Highlighted by City)",
    legend_title="City (Outliers)",
    xaxis_title="Total Revenue",
    yaxis_title="Total Income",
    template="plotly_white",
    height=700
)

fig.show()
fig.write_html("scatterplot_outliers.html")



# Create output directories if they don't exist
os.makedirs("outlier_names", exist_ok=True)
os.makedirs("city_plots", exist_ok=True)

def save_outliers_and_plot(city_name, df):
    """Save outlier names and generate interactive plot for a city."""
    outliers = df[df['outlier'] == True]
    
    # Save outlier names to txt file
    outlier_names = outliers['name'].dropna().unique()
    with open(f"outlier_names/{city_name.lower().replace(' ', '_')}_outliers.txt", "w") as f:
        for name in outlier_names:
            f.write(f"{name}\n")
    
    # Generate interactive plot
    fig = px.scatter(
        df[df['outlier'] == False],
        x='totrev',
        y='total_income',
        opacity=0.4,
        color_discrete_sequence=['gray'],
        hover_data=['name', 'city', 'totrev', 'total_income'],
        labels={'totrev': 'Total Revenue', 'total_income': 'Total Income'},
    )

    outlier_fig = px.scatter(
        outliers,
        x='totrev',
        y='total_income',
        color='city',
        hover_data=['name', 'city', 'totrev', 'total_income'],
        labels={'totrev': 'Total Revenue', 'total_income': 'Total Income'},
    )

    for trace in outlier_fig.data:
        fig.add_trace(trace)

    fig.update_layout(
        title=f"{city_name} Outliers: Total Revenue vs Total Income",
        legend_title="City (Outliers)",
        template="plotly_white"
    )

    fig.write_html(f"city_plots/{city_name.lower().replace(' ', '_')}_plot.html")

# Map cities to their DataFrames
city_dfs = {
    "New York": new_york,
    "Atlanta": atlanta,
    "Chicago": chicago,
    "Houston": houston,
    "Dallas": dallas,
    "Washington": washington,
    "Los Angeles": losangeles,
    "Las Vegas": lasvegas
}

# Loop through each city and process
for city_name, df in city_dfs.items():
    save_outliers_and_plot(city_name, df)


import plotly.express as px

# Exclude NEW YORK and filter to non-outliers
non_ny_non_outliers = total[
    (total['city'] != "NEW YORK") & (total['outlier'] == False)
]

# Create the scatter plot
fig = px.scatter(
    non_ny_non_outliers,
    x='totrev',
    y='total_income',
    color='city',
    hover_data=['name', 'city', 'totrev', 'total_income'],
    labels={'totrev': 'Total Revenue', 'total_income': 'Total Income'},
    title="Interactive Scatterplot (Excludes New York): Total Revenue vs Total Income (Non-Outliers Only)",
    template="plotly_white"
)

# Final layout
fig.update_layout(
    height=700,
    legend_title="City",
    xaxis_title="Total Revenue",
    yaxis_title="Total Income",
)

# Display and save
fig.show()
fig.write_html("city_plots/non_ny_non_outlier_plot.html")
