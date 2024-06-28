import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from scipy.stats import f_oneway
from scipy.stats import kruskal
from scipy.stats import pearsonr, spearmanr
import re
from scipy.stats import ttest_ind, f_oneway
import os
import inspect


IMAGE_DIR = r'C:\Users\milan\Documents\projects\booking-com review-analysis\images'

def get_cleaned_tags(tags):
    # Define the tags to extract with regex patterns
    tags_patterns = {
        'Leisure': r'leisure|leisure trip|leisure travel',
        'Solo': r'solo|solo traveler',
        'Family': r'family|family with young children',
        'Business': r'business|business trip',
        'Couple': r'couple'
    }

    # Function to extract and create new columns for specific tags
    tags = tags.strip("[]").lower()  # Clean and convert tags to lower case
    tag_dict = {tag: 0 for tag in tags_patterns}  # Initialize dictionary with all tags set to 0
    for tag_name, tag_pattern in tags_patterns.items():
        if re.search(tag_pattern, tags):
            tag_dict[tag_name] = 1  # Set to 1 if pattern matches
    return tuple(tag_dict[tag] for tag in tags_patterns)




def show_review_count_by_nat(df):
    nationality_counts = df['Reviewer_Nationality'].value_counts().reset_index()
    nationality_counts.columns = ['Reviewer_Nationality', 'Number_of_Reviews']

    # Plot the data
    plt.figure(figsize=(15, 8))
    sns.barplot(x='Number_of_Reviews', y='Reviewer_Nationality', data=nationality_counts.head(20))
    plt.title('Number of Reviews by Nationality (Top 20)')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Reviewer Nationality')
    plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, '{0}.png'.format(inspect.stack()[0][3])))

def reviews_by_date(df):
    # Convert the Review_Date column to datetime


    # Count the number of reviews by date for the UK
    df_by_date = df['Review_Date'].value_counts().reset_index()
    df_by_date.columns = ['Review_Date', 'Number_of_Reviews']
    df_by_date = df_by_date.sort_values('Review_Date')

    # Plot the data
    plt.figure(figsize=(15, 8))
    sns.lineplot(x='Review_Date', y='Number_of_Reviews', data=df_by_date)
    plt.title('Number of Reviews by Date for UK Reviewers')
    plt.xlabel('Date')
    plt.ylabel('Number of Reviews')
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, '{0}.png'.format(inspect.stack()[0][3])))

def reviews_by_date_freq(data):
    # Calculate number of reviews per year
    reviews_per_year = data['Year'].value_counts().reset_index()
    reviews_per_year.columns = ['Year', 'Number_of_Reviews']
    reviews_per_year = reviews_per_year.sort_values('Year')

    # Calculate number of reviews per month
    reviews_per_month = data['Month'].value_counts().reset_index()
    reviews_per_month.columns = ['Month', 'Number_of_Reviews']
    reviews_per_month = reviews_per_month.sort_values('Month')

    # Calculate number of reviews per day of the week
    reviews_per_day_of_week = data['Day_of_Week'].value_counts().reset_index()
    reviews_per_day_of_week.columns = ['Day_of_Week', 'Number_of_Reviews']
    reviews_per_day_of_week = reviews_per_day_of_week.sort_values('Number_of_Reviews', ascending=False)

    # Calculate number of reviews per day of the month
    reviews_per_day_of_month = data['Day_of_Month'].value_counts().reset_index()
    reviews_per_day_of_month.columns = ['Day_of_Month', 'Number_of_Reviews']
    reviews_per_day_of_month = reviews_per_day_of_month.sort_values('Day_of_Month')

    # Calculate number of reviews for weekends vs weekdays
    reviews_weekend = data['Weekend'].value_counts().reset_index()
    reviews_weekend.columns = ['Weekend', 'Number_of_Reviews']
    reviews_weekend['Weekend'] = reviews_weekend['Weekend'].map({True: 'Weekend', False: 'Weekday'})

    # Plot yearly reviews
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Year', y='Number_of_Reviews', data=reviews_per_year)
    plt.title('Number of Reviews per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Reviews')
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, '{0}_year.png'.format(inspect.stack()[0][3])))

    # Plot monthly reviews
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Month', y='Number_of_Reviews', data=reviews_per_month)
    plt.title('Number of Reviews per Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Reviews')
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, '{0}_month.png'.format(inspect.stack()[0][3])))

    # Plot day of the week reviews
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Number_of_Reviews', y='Day_of_Week', data=reviews_per_day_of_week)
    plt.title('Number of Reviews per Day of the Week')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Day of the Week')
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, '{0}_dow.png'.format(inspect.stack()[0][3])))

    # Plot day of the month reviews
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Day_of_Month', y='Number_of_Reviews', data=reviews_per_day_of_month)
    plt.title('Number of Reviews per Day of the Month')
    plt.xlabel('Day of the Month')
    plt.ylabel('Number of Reviews')
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, '{0}_dom.png'.format(inspect.stack()[0][3])))

    # Plot weekday vs weekend reviews
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Number_of_Reviews', y='Weekend', data=reviews_weekend)
    plt.title('Number of Reviews: Weekday vs Weekend')
    plt.xlabel('Number of Reviews')
    plt.ylabel('')
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, '{0}_weekday.png'.format(inspect.stack()[0][3])))

def box_plot_num_reviews_nat(data):
    top_nationalities = data['Reviewer_Nationality'].value_counts().head(10).index

    # Filter the data for only these top 10 nationalities
    top_nationalities_data = data[data['Reviewer_Nationality'].isin(top_nationalities)]

    # Create a box plot of Reviewer Nationality and Total Number of Reviews by Reviewer
    plt.figure(figsize=(15, 8))
    sns.boxplot(x='Reviewer_Nationality', y='Total_Number_of_Reviews_Reviewer_Has_Given', data=top_nationalities_data)
    plt.title('Box Plot of Reviewer Nationality and Total Number of Reviews by Reviewer')
    plt.xlabel('Reviewer Nationality')
    plt.ylabel('Total Number of Reviews by Reviewer')
    plt.xticks(rotation=45)
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, '{0}.png'.format(inspect.stack()[0][3])))

def box_plot_rev_score_nat(data):
    top_nationalities = data['Reviewer_Nationality'].value_counts().head(10).index

    # Filter the data for only these top 10 nationalities
    top_nationalities_data = data[data['Reviewer_Nationality'].isin(top_nationalities)]

    # Create a box plot of Reviewer Nationality and Total Number of Reviews by Reviewer
    plt.figure(figsize=(15, 8))
    sns.boxplot(x='Reviewer_Nationality', y='Reviewer_Score', data=top_nationalities_data)
    plt.title('Box Plot of Reviewer Nationality and Review Score by Reviewer')
    plt.xlabel('Reviewer Nationality')
    plt.xticks(rotation=45)
    plt.ylabel('Review Score by Reviewer')
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, '{0}.png'.format(inspect.stack()[0][3])))


def hotels_by_num_reviews(data):
    hotel_review_counts = data['Hotel_Name'].value_counts().reset_index()
    hotel_review_counts.columns = ['Hotel_Name', 'Total_Number_of_Reviews']

    # Get the top 10 hotels
    top_10_hotels = hotel_review_counts.head(10)
    # Visualize the top 10 hotels by total number of reviews
    plt.figure(figsize=(15, 8))
    sns.barplot(x='Total_Number_of_Reviews', y='Hotel_Name', data=top_10_hotels)
    plt.title('Top 10 Hotels by Total Number of Reviews')
    plt.xlabel('Total Number of Reviews')
    plt.ylabel('Hotel Name')
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, '{0}.png'.format(inspect.stack()[0][3])))


def top_cities(df):
    # Calculate the number of reviews per city
    city_review_counts = df['City'].value_counts().head(10)

    # Create a bar plot using seaborn
    plt.figure(figsize=(12, 8))
    sns.barplot(x=city_review_counts.index, y=city_review_counts.values)
    plt.title('Top Cities by Number of Reviews')
    plt.xlabel('City')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=45)
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, '{0}.png'.format(inspect.stack()[0][3])))

def city_and_rev_nat(df):
    top_cities = df['City'].value_counts().head(10).index

    # Prepare data for sunburst plot
    sunburst_data = df[df['City'].isin(top_cities)].groupby(['City', 'Reviewer_Nationality']).size().reset_index(
        name='Counts')

    # Create the sunburst plot with city on the outside and reviewer nationality on the inside
    fig = px.sunburst(sunburst_data, path=['Reviewer_Nationality', 'City'], values='Counts',
                      color='Counts', color_continuous_scale='viridis',
                      title='Top Nationalities Reviews in Top Cities')

    # Update layout for better presentation
    fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))

    # Show the plot
    fig.show()



def city_num_review_by_dow_diff_plots(data):
    df = data.copy()

    # Extract the day of the week from Review_Date
    df['Review_DayOfWeek'] = pd.to_datetime(df['Review_Date']).dt.dayofweek

    # Get the top 20 cities by the number of reviews
    top_20_cities = df['City'].value_counts().head(20).index

    # Create a pivot table with City as rows and Review_DayOfWeek as columns
    heatmap_data_week_raw = df[df['City'].isin(top_20_cities)].pivot_table(index='City', columns='Review_DayOfWeek',
                                                                           aggfunc='size', fill_value=0)

    # Create a subplot for each city
    fig, axes = plt.subplots(nrows=len(top_20_cities), ncols=1, figsize=(20, 15), sharex=True, sharey=False)

    # Iterate over the axes and top 20 cities to create individual heatmaps
    for ax, city in zip(axes.flatten(), top_20_cities):
        city_data = heatmap_data_week_raw.loc[city].values.reshape(1, -1)
        sns.heatmap(city_data, ax=ax, cmap='Blues', cbar=False, annot=True, fmt='d', linewidths=.5, linecolor='gray',
                    annot_kws={"size": 14})
        ax.set_xticks(np.arange(7) + 0.5)
        ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=45)
        ax.set_yticks([])
        ax.set_ylabel(city, rotation=0, labelpad=60, fontsize=12, weight='bold', ha='right')

    # Adjust layout
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, '{0}.png'.format(inspect.stack()[0][3])))


def city_num_review_by_dom_diff_plots(data):
    df = data.copy()
    # Extract the day of the month from Review_Date
    df['Review_Day'] = pd.to_datetime(df['Review_Date']).dt.day

    # Get the top 20 cities by the number of reviews
    top_20_cities = df['City'].value_counts().head(20).index

    # Create a pivot table with City as rows and Review_Day as columns
    heatmap_data_day = df[df['City'].isin(top_20_cities)].pivot_table(index='City', columns='Review_Day',
                                                                      aggfunc='size', fill_value=0)

    # Reindex to ensure all days from 1 to 31 are represented
    heatmap_data_day = heatmap_data_day.reindex(columns=np.arange(1, 32), fill_value=0)

    # Create a subplot for each city
    fig, axes = plt.subplots(nrows=len(top_20_cities), ncols=1, figsize=(20, 15), sharex=True, sharey=True)

    # Iterate over the axes and top 20 cities to create individual heatmaps
    for ax, city in zip(axes.flatten(), top_20_cities):
        city_data = heatmap_data_day.loc[city].values.reshape(1, -1)
        sns.heatmap(city_data, ax=ax, cmap='Blues', cbar=False, fmt='d', linewidths=.5, linecolor='gray')
        ax.set_yticks([])
        ax.set_ylabel(city, rotation=0, labelpad=60, fontsize=12, weight='bold', ha='right')
        ax.set_xticks(np.arange(31) + 0.5)
        ax.set_xticklabels(np.arange(1, 32), rotation=45)

    # Adjust layout
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, '{0}.png'.format(inspect.stack()[0][3])))


def rating_by_city(data):
    df = data.copy()
    df = df.dropna(subset=['Reviewer_Score'])
    # Extract the year and month from Review_Date
    df['YearMonth'] = df['Review_Date'].dt.strftime('%Y-%b')  # Extract YearMonth as a string in 'YYYY-MMM' format

    # Calculate the average rating for each city over years and months
    median_rating = df.groupby(['YearMonth', 'City'])['Reviewer_Score'].median().reset_index()

    # Get the top 20 cities by the number of reviews
    top_20_cities = df['City'].value_counts().head(20).index

    # Filter data for the top 20 cities
    median_rating_top_cities = median_rating[median_rating['City'].isin(top_20_cities)]
    median_rating_top_cities['Reviewer_Score'] = median_rating_top_cities['Reviewer_Score'].astype(int)

    # Create subplots for each city
    fig, axes = plt.subplots(nrows=len(top_20_cities), ncols=1, figsize=(20, 15), sharex=True, sharey=True)

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Plot each city's data in a separate subplot
    for ax, city in zip(axes, top_20_cities):
        city_data = median_rating_top_cities[median_rating_top_cities['City'] == city]
        sns.lineplot(data=city_data, x='YearMonth', y='Reviewer_Score', marker='o', ax=ax)
        ax.set_ylabel(city, rotation=45, labelpad=60, fontsize=12, ha='right')
        ax.tick_params(axis='x', rotation=45)

    # Adjust layout
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, '{0}.png'.format(inspect.stack()[0][3])))


def anova_nationality_reviewer_score(data):
    df = data.copy()
    # Clean the Reviewer_Score column
    df['Reviewer_Score'] = pd.to_numeric(df['Reviewer_Score'],
                                         errors='coerce')  # Convert to numeric, setting errors to NaN
    df = df.dropna(subset=['Reviewer_Score'])  # Drop rows with NaN values in Reviewer_Score

    # Get the top nationalities by number of reviews (to reduce the number of groups in the ANOVA test)
    top_nationalities = df['Reviewer_Nationality'].value_counts().head(20).index

    # Filter data for the top nationalities
    df_top_nationalities = df[df['Reviewer_Nationality'].isin(top_nationalities)]

    scores_by_nationality = []
    for _nationality in top_nationalities:
        _data = df_top_nationalities[df_top_nationalities['Reviewer_Nationality'] == _nationality]['Reviewer_Score']
        scores_by_nationality.append(_data)

    # Perform the ANOVA test
    anova_result = f_oneway(*scores_by_nationality)

    # Output the results
    print('ANOVA test result:', anova_result)

    # Interpretation
    if anova_result.pvalue < 0.05:
        print(
            "The p-value is less than 0.05 with a value of {0}, indicating significant differences in review scores between different nationalities.".format(anova_result.pvalue))
    else:
        print(
            "The p-value is greater than or equal to 0.05, indicating no significant differences in review scores between different nationalities.")

    plt.figure(figsize=(14, 8))
    sns.violinplot(data=df_top_nationalities, x='Reviewer_Nationality', y='Reviewer_Score', palette='Set3')
    plt.title('Distribution and Density of Review Scores by Nationality')
    plt.xlabel('Nationality')
    plt.ylabel('Review Score')
    plt.xticks(rotation=45)
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, '{0}.png'.format(inspect.stack()[0][3])))
    '''
    Specific Observations:
    Consistent High Scores:
    
    Countries like Belgium, Saudi Arabia, and United Arab Emirates have a higher density of scores around 9 and 10, indicating that reviewers from these nationalities tend to give higher scores more consistently.
    Variable Scores:
    
    Countries like United States of America and Germany show a wider spread of review scores, indicating a more varied opinion among reviewers from these nationalities.
    Lower Scores:
    
    Reviewers from countries like South Africa and Israel have distributions that extend towards the lower end (below 5), indicating that there are more low scores given by reviewers from these nationalities.
    Middle Range Scores:
    
    Nationalities like Ireland and France have more scores clustered around the median range (6-8), indicating a tendency to give moderate reviews.
    '''


def city_sentiment_test(data):

    df = data.copy()
    # Ensure that the sentiment columns are numeric
    df['Positive_Review_Sentiment'] = pd.to_numeric(df['Positive_Review_Sentiment'], errors='coerce')
    df['Negative_Review_Sentiment'] = pd.to_numeric(df['Negative_Review_Sentiment'], errors='coerce')

    # Drop rows with missing sentiment data
    df = df.dropna(subset=['Positive_Review_Sentiment', 'Negative_Review_Sentiment'])

    # Get the top cities by the number of reviews
    top_cities = df['City'].value_counts().head(20).index

    # Filter data for the top cities
    df_top_cities = df[df['City'].isin(top_cities)]

    # Group Positive_Review_Sentiment by City
    positive_sentiment_by_city = [df_top_cities[df_top_cities['City'] == city]['Positive_Review_Sentiment'] for city in
                                  top_cities]

    # Perform Kruskal-Wallis test
    kruskal_result_positive = kruskal(*positive_sentiment_by_city)
    print(
        f"Kruskal-Wallis test for Positive Review Sentiment: H-statistic = {kruskal_result_positive.statistic}, p-value = {kruskal_result_positive.pvalue}")

    # Group Negative_Review_Sentiment by City
    negative_sentiment_by_city = [df_top_cities[df_top_cities['City'] == city]['Negative_Review_Sentiment'] for city in
                                  top_cities]

    # Perform Kruskal-Wallis test
    kruskal_result_negative = kruskal(*negative_sentiment_by_city)
    print(
        f"Kruskal-Wallis test for Negative Review Sentiment: H-statistic = {kruskal_result_negative.statistic}, p-value = {kruskal_result_negative.pvalue}")

    # Create box plots for Positive Review Sentiment
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df_top_cities, x='City', y='Positive_Review_Sentiment', palette='Set3')
    plt.title('Distribution of Positive Review Sentiments by City')
    plt.xlabel('City')
    plt.ylabel('Positive Review Sentiment')
    plt.xticks(rotation=45)
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, '{0}_positive.png'.format(inspect.stack()[0][3])))

    # Create box plots for Negative Review Sentiment
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df_top_cities, x='City', y='Negative_Review_Sentiment', palette='Set2')
    plt.title('Distribution of Negative Review Sentiments by City')
    plt.xlabel('City')
    plt.ylabel('Negative Review Sentiment')
    plt.xticks(rotation=45)
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, '{0}_negative.png'.format(inspect.stack()[0][3])))

    '''
    The box plot visually supports the hypothesis that there are significant differences in negative review sentiments between cities. The differences in median values, IQR, and the range of sentiments indicate that reviewers' negative sentiments vary based on the city being reviewed.

    Combined Analysis of Positive and Negative Sentiments:
    Positive Sentiments:
    
    More variability and outliers in positive sentiments indicate that some cities receive a wide range of positive feedback.
    Negative Sentiments:
    
    Less variability and fewer outliers in negative sentiments suggest that negative feedback is more consistent across cities.'''


def review_length_sentiment_test(data):
    df = data.copy()
    # Ensure that the word count and sentiment columns are numeric
    df['Review_Total_Positive_Word_Counts'] = pd.to_numeric(df['Review_Total_Positive_Word_Counts'], errors='coerce')
    df['Positive_Review_Sentiment'] = pd.to_numeric(df['Positive_Review_Sentiment'], errors='coerce')
    df['Review_Total_Negative_Word_Counts'] = pd.to_numeric(df['Review_Total_Negative_Word_Counts'], errors='coerce')
    df['Negative_Review_Sentiment'] = pd.to_numeric(df['Negative_Review_Sentiment'], errors='coerce')

    # Drop rows with missing data in the relevant columns
    df = df.dropna(
        subset=['Review_Total_Positive_Word_Counts', 'Positive_Review_Sentiment', 'Review_Total_Negative_Word_Counts',
                'Negative_Review_Sentiment'])

    # Calculate Pearson and Spearman correlations for positive reviews
    pearson_corr_pos, _ = pearsonr(df['Review_Total_Positive_Word_Counts'], df['Positive_Review_Sentiment'])
    spearman_corr_pos, _ = spearmanr(df['Review_Total_Positive_Word_Counts'], df['Positive_Review_Sentiment'])

    # Calculate Pearson and Spearman correlations for negative reviews
    pearson_corr_neg, _ = pearsonr(df['Review_Total_Negative_Word_Counts'], df['Negative_Review_Sentiment'])
    spearman_corr_neg, _ = spearmanr(df['Review_Total_Negative_Word_Counts'], df['Negative_Review_Sentiment'])

    print(
        f"Positive Review Sentiment - Pearson Correlation: {pearson_corr_pos}, Spearman Correlation: {spearman_corr_pos}")
    print(
        f"Negative Review Sentiment - Pearson Correlation: {pearson_corr_neg}, Spearman Correlation: {spearman_corr_neg}")

    # Scatter plot for positive review sentiment vs. word count
    plt.figure(figsize=(14, 6))
    sns.scatterplot(data=df, x='Review_Total_Positive_Word_Counts', y='Positive_Review_Sentiment')
    plt.title('Positive Review Sentiment vs. Positive Word Count')
    plt.xlabel('Total Positive Word Count')
    plt.ylabel('Positive Review Sentiment')
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, '{0}_positive.png'.format(inspect.stack()[0][3])))

    # Scatter plot for negative review sentiment vs. word count
    plt.figure(figsize=(14, 6))
    sns.scatterplot(data=df, x='Review_Total_Negative_Word_Counts', y='Negative_Review_Sentiment')
    plt.title('Negative Review Sentiment vs. Negative Word Count')
    plt.xlabel('Total Negative Word Count')
    plt.ylabel('Negative Review Sentiment')
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, '{0}_negative.png'.format(inspect.stack()[0][3])))


def review_length_sentiment_test_by_nat(data):
    df = data.copy()
    # Ensure that the word count and sentiment columns are numeric
    df['Review_Total_Positive_Word_Counts'] = pd.to_numeric(df['Review_Total_Positive_Word_Counts'], errors='coerce')
    df['Positive_Review_Sentiment'] = pd.to_numeric(df['Positive_Review_Sentiment'], errors='coerce')
    df['Review_Total_Negative_Word_Counts'] = pd.to_numeric(df['Review_Total_Negative_Word_Counts'], errors='coerce')
    df['Negative_Review_Sentiment'] = pd.to_numeric(df['Negative_Review_Sentiment'], errors='coerce')

    # Drop rows with missing data in the relevant columns
    df = df.dropna(
        subset=['Review_Total_Positive_Word_Counts', 'Positive_Review_Sentiment', 'Review_Total_Negative_Word_Counts',
                'Negative_Review_Sentiment'])

    # Get the top nationalities by the number of reviews
    top_nationalities = df['Reviewer_Nationality'].value_counts().head(10).index

    # Initialize a dictionary to store correlation results
    correlation_results = {}

    # Calculate and store correlation for each nationality
    for nationality in top_nationalities:
        subset = df[df['Reviewer_Nationality'] == nationality]

        # Calculate Pearson and Spearman correlations for positive reviews
        pearson_corr_pos, _ = pearsonr(subset['Review_Total_Positive_Word_Counts'], subset['Positive_Review_Sentiment'])
        spearman_corr_pos, _ = spearmanr(subset['Review_Total_Positive_Word_Counts'],
                                         subset['Positive_Review_Sentiment'])

        # Calculate Pearson and Spearman correlations for negative reviews
        pearson_corr_neg, _ = pearsonr(subset['Review_Total_Negative_Word_Counts'], subset['Negative_Review_Sentiment'])
        spearman_corr_neg, _ = spearmanr(subset['Review_Total_Negative_Word_Counts'],
                                         subset['Negative_Review_Sentiment'])

        # Store the results
        correlation_results[nationality] = {
            'pearson_pos': pearson_corr_pos,
            'spearman_pos': spearman_corr_pos,
            'pearson_neg': pearson_corr_neg,
            'spearman_neg': spearman_corr_neg
        }

        # Create scatter plots for positive and negative sentiments
        plt.figure(figsize=(14, 6))
        sns.scatterplot(data=subset, x='Review_Total_Positive_Word_Counts', y='Positive_Review_Sentiment')
        plt.title(f'Positive Review Sentiment vs. Positive Word Count for {nationality}')
        plt.xlabel('Total Positive Word Count')
        plt.ylabel('Positive Review Sentiment')
        # plt.show()
        plt.savefig(os.path.join(IMAGE_DIR, '{0}_positive.png'.format(inspect.stack()[0][3])))

        plt.figure(figsize=(14, 6))
        sns.scatterplot(data=subset, x='Review_Total_Negative_Word_Counts', y='Negative_Review_Sentiment')
        plt.title(f'Negative Review Sentiment vs. Negative Word Count for {nationality}')
        plt.xlabel('Total Negative Word Count')
        plt.ylabel('Negative Review Sentiment')
        # plt.show()
        plt.savefig(os.path.join(IMAGE_DIR, '{0}_negative.png'.format(inspect.stack()[0][3])))

    # Display the correlation results
    for nationality, results in correlation_results.items():
        print(f"{nationality}:")
        print(f"  Positive Sentiment - Pearson: {results['pearson_pos']:.4f}, Spearman: {results['spearman_pos']:.4f}")
        print(f"  Negative Sentiment - Pearson: {results['pearson_neg']:.4f}, Spearman: {results['spearman_neg']:.4f}")
        print()


def review_length_test_by_nationality(data):
    df = data.copy()
    # Clean the data: Remove rows with NaN values in important columns
    df = df.dropna(subset=['Reviewer_Nationality', 'Negative_Review', 'Positive_Review'])

    # Calculate the length of each review (negative and positive)
    df['Negative_Review_Length'] = df['Negative_Review'].apply(len)
    df['Positive_Review_Length'] = df['Positive_Review'].apply(len)

    # Get the top nationalities by the number of reviews (to simplify the analysis)
    top_nationalities = df['Reviewer_Nationality'].value_counts().head(20).index

    # Filter data for the top nationalities
    df_top_nationalities = df[df['Reviewer_Nationality'].isin(top_nationalities)]

    # Encode nationalities as numerical values
    df_top_nationalities['Nationality_Code'] = df_top_nationalities['Reviewer_Nationality'].astype('category').cat.codes

    # Calculate the average review length for each nationality
    avg_neg_review_length_by_nationality = df_top_nationalities.groupby('Reviewer_Nationality')[
        'Negative_Review_Length'].mean()
    avg_pos_review_length_by_nationality = df_top_nationalities.groupby('Reviewer_Nationality')[
        'Positive_Review_Length'].mean()

    # Calculate the correlation between nationality (as a numerical code) and review lengths
    correlation_neg, p_value_neg = spearmanr(df_top_nationalities['Nationality_Code'],
                                             df_top_nationalities['Negative_Review_Length'])
    correlation_pos, p_value_pos = spearmanr(df_top_nationalities['Nationality_Code'],
                                             df_top_nationalities['Positive_Review_Length'])

    # Output the results
    print(
        f'Correlation between nationality and negative review length: {correlation_neg:.3f} (p-value: {p_value_neg:.3f})')
    print(
        f'Correlation between nationality and positive review length: {correlation_pos:.3f} (p-value: {p_value_pos:.3f})')

    # Plot the average review length by nationality
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 14))

    sns.barplot(x=avg_neg_review_length_by_nationality.index, y=avg_neg_review_length_by_nationality.values,
                palette='viridis', ax=axes[0])
    axes[0].set_xticklabels(avg_neg_review_length_by_nationality.index, rotation=90)
    axes[0].set_xlabel('Nationality')
    axes[0].set_ylabel('Average Negative Review Length')
    axes[0].set_title('Average Negative Review Length by Nationality')

    sns.barplot(x=avg_pos_review_length_by_nationality.index, y=avg_pos_review_length_by_nationality.values,
                palette='viridis', ax=axes[1])
    axes[1].set_xticklabels(avg_pos_review_length_by_nationality.index, rotation=90)
    axes[1].set_xlabel('Nationality')
    axes[1].set_ylabel('Average Positive Review Length')
    axes[1].set_title('Average Positive Review Length by Nationality')

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, '{0}.png'.format(inspect.stack()[0][3])))


def average_score_vs_individual_score(data):

    df = data.copy()
    # Clean the data: Remove rows with NaN values in important columns
    df = df.dropna(subset=['Average_Score', 'Reviewer_Score'])

    # Ensure the columns are numeric
    df['Average_Score'] = pd.to_numeric(df['Average_Score'], errors='coerce')
    df['Reviewer_Score'] = pd.to_numeric(df['Reviewer_Score'], errors='coerce')

    # Calculate the correlation
    correlation, p_value = pearsonr(df['Average_Score'], df['Reviewer_Score'])

    # Output the results
    print(f'Correlation between Average_Score and Reviewer_Score: {correlation:.3f}')
    print(f'P-value: {p_value:.3f}')

    # Interpretation
    if p_value < 0.05:
        print("The p-value is less than 0.05, indicating a statistically significant correlation.")
        if correlation > 0:
            print(
                "The positive correlation suggests that individual reviewers are positively influenced by the average score.")
        else:
            print(
                "The correlation is negative, suggesting that individual reviewers are negatively influenced by the average score.")
    else:
        print("The p-value is greater than or equal to 0.05, indicating no statistically significant correlation.")


def anova_trip_types(data):

    df = data.copy()
    # Ensure sentiment columns are numeric
    df['Positive_Review_Sentiment'] = pd.to_numeric(df['Positive_Review_Sentiment'], errors='coerce')
    df['Negative_Review_Sentiment'] = pd.to_numeric(df['Negative_Review_Sentiment'], errors='coerce')

    # Function to perform t-test for two groups
    def perform_t_test(group1, group2, sentiment):
        t_stat, p_value = ttest_ind(group1[sentiment], group2[sentiment], nan_policy='omit')
        return t_stat, p_value

    # Function to perform ANOVA for multiple groups
    def perform_anova(groups):
        f_stat, p_value = f_oneway(*groups)
        return f_stat, p_value

    # List of trip types
    trip_types = ['leisure', 'solo', 'family', 'business', 'couple']

    # Perform analysis for Positive_Review_Sentiment
    positive_sentiments = [df[df[trip_type] == 1]['Positive_Review_Sentiment'].dropna() for trip_type in trip_types]
    f_stat_pos, p_value_pos = perform_anova(positive_sentiments)

    # Perform analysis for Negative_Review_Sentiment
    negative_sentiments = [df[df[trip_type] == 1]['Negative_Review_Sentiment'].dropna() for trip_type in trip_types]
    f_stat_neg, p_value_neg = perform_anova(negative_sentiments)

    # Output the results
    print(
        f'ANOVA for Positive Review Sentiment by Trip Type: F-statistic = {f_stat_pos:.3f}, p-value = {p_value_pos:.3f}')
    print(
        f'ANOVA for Negative Review Sentiment by Trip Type: F-statistic = {f_stat_neg:.3f}, p-value = {p_value_neg:.3f}')

    # Interpretation for Positive Review Sentiment
    if p_value_pos < 0.05:
        print("There are significant differences in positive review sentiment based on trip type.")
    else:
        print("There are no significant differences in positive review sentiment based on trip type.")

    # Interpretation for Negative Review Sentiment
    if p_value_neg < 0.05:
        print("There are significant differences in negative review sentiment based on trip type.")
    else:
        print("There are no significant differences in negative review sentiment based on trip type.")

    # Melt the DataFrame to have a 'Trip_Type' column and a 'Sentiment' column
    df_melted = pd.melt(df, id_vars=['Positive_Review_Sentiment', 'Negative_Review_Sentiment'],
                        value_vars=['leisure', 'solo', 'family', 'business', 'couple'],
                        var_name='Trip_Type', value_name='Included')

    # Filter rows where 'Included' is 1
    df_melted = df_melted[df_melted['Included'] == 1]

    # Create a separate DataFrame for positive and negative sentiments
    df_positive = df_melted[['Trip_Type', 'Positive_Review_Sentiment']].dropna()
    df_negative = df_melted[['Trip_Type', 'Negative_Review_Sentiment']].dropna()

    # Create box plots
    plt.figure(figsize=(14, 7))

    # Box plot for Positive Review Sentiment
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Trip_Type', y='Positive_Review_Sentiment', data=df_positive, palette='viridis')
    plt.title('Positive Review Sentiment by Trip Type')
    plt.xlabel('Trip Type')
    plt.ylabel('Positive Review Sentiment')

    # Box plot for Negative Review Sentiment
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Trip_Type', y='Negative_Review_Sentiment', data=df_negative, palette='viridis')
    plt.title('Negative Review Sentiment by Trip Type')
    plt.xlabel('Trip Type')
    plt.ylabel('Negative Review Sentiment')

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, '{0}.png'.format(inspect.stack()[0][3])))


if __name__ == "__main__":
    filename = r'D:\projects\datasets\booking-com-reviews2-europe\Hotel_Reviews_with_sentiment.csv'
    location_file = r'D:\projects\datasets\booking-com-reviews2-europe\city_data.csv'

    location_df = pd.read_csv(location_file)
    df = pd.read_csv(filename)

    df = pd.merge(df, location_df, on=['lat', 'lng'], how='left')

    print(df.head())
    df['Review_Date'] = pd.to_datetime(df['Review_Date'], errors='coerce')
    # Extract additional date-related features
    df['Year'] = df['Review_Date'].dt.year
    df['Month'] = df['Review_Date'].dt.month
    df['Day_of_Week'] = df['Review_Date'].dt.day_name()
    df['Day_of_Month'] = df['Review_Date'].dt.day
    df['Weekend'] = df['Review_Date'].dt.dayofweek >= 5

    df['City'] = df['City'].replace(['City of London'], 'London')

    lambda_func = lambda x: get_cleaned_tags(x['Tags'])
    df['tag_temp'] = df.apply(lambda_func, axis=1)
    tag_list = ['leisure', 'solo', 'family', 'business', 'couple']
    for n, col in enumerate(tag_list):
        df[col] = df['tag_temp'].apply(lambda tag_temp: tag_temp[n])

    df = df.drop('tag_temp', axis=1)


    number_of_hotels = len(df.Hotel_Name.unique())
    number_of_dates = len(df.Review_Date.unique())
    number_of_rev_nat = len(df.Reviewer_Nationality.unique())

    print(df.head())

    print('number_of_hotels', number_of_hotels)
    print('number_of_dates', number_of_dates)
    print('number_of_rev_nat', number_of_rev_nat)

    reviews_by_date(df)
    reviews_by_date_freq(df)
    box_plot_num_reviews_nat(df)
    box_plot_rev_score_nat(df)
    hotels_by_num_reviews(df)
    top_cities(df)
    city_num_review_by_dow_diff_plots(df)
    city_num_review_by_dom_diff_plots(df)
    rating_by_city(df)
    #
    anova_nationality_reviewer_score(df)
    city_sentiment_test(df)
    review_length_sentiment_test(df)
    review_length_sentiment_test_by_nat(df)
    review_length_test_by_nationality(df)
    #
    average_score_vs_individual_score(df)
    anova_trip_types(df)
