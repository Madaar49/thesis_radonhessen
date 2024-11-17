# -*- coding: utf-8 -*-
"""
__author__ = "Augustine M Gbondo"
__credits__ = ["Augustine M Gbondo"]
__project__ = Geogenic radon potential mapping in Hessen using machine learning techniques
__maintainer__ = "Augustine M Gbondo"
__email__ = "gbondomadaar@gmail.com"
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, shape, box
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
from plotnine import ggplot, aes, geom_tile, geom_text, labs, theme, element_text, scale_fill_cmap
import geopandas as gpd
from shapely.geometry import Point


class GeodataProcessor:
    def __init__(self, crs="EPSG:4326"):
        """
        Initialize GeodataProcessor with default CRS.
        """
        self.crs = crs

    def compute_area(self, gdf, field="area", crs=None):
        """Compute area in square kilometers of geometries in GeoDataFrame"""
        crs = crs if crs else self.crs
        gdf = gdf.to_crs(crs)
        gdf[field] = gdf.geometry.area / 10**6
        return gdf

    @staticmethod
    def convert_to_geodataframe(df, latitude, longitude):
        """
        Convert a DataFrame to GeoDataFrame with EPSG:4326 CRS.
        """
        geometry = [Point(xy) for xy in zip(df[longitude], df[latitude])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
        gdf.set_crs(epsg=4326, inplace=True)
        return gdf
    
    def create_geodataframe(self, data, crs=None):
        """Create a GeoDataFrame"""
        crs = crs if crs else self.crs
        return gpd.GeoDataFrame(data, geometry=data["geometry"], crs=crs)

    @staticmethod
    def create_shapely_box(geom):
        """Create a bounding box using shapely"""
        minx, miny, maxx, maxy = geom.bounds
        return box(minx, miny, maxx, maxy)

    def create_tiles(self, gdf, width=1, length=1, file=None):
        """Create vector tiles from the total bounds of a GeoDataFrame"""
        xmin, ymin, xmax, ymax = self.geodataframe_bounds(gdf)
        cols = np.arange(xmin, xmax + width, width)
        rows = np.arange(ymin, ymax + length, length)

        polygons = [Polygon([(x, y), (x + width, y), (x + width, y + length), (x, y + length)])
                    for x in cols[:-1] for y in rows[:-1]]

        grid = self.create_geodataframe({'geometry': polygons})
        if file:
            grid.to_file(file)
        return grid

    @staticmethod
    def drop_columns_dataframe(gdf, columns):
        """Drop specified columns from GeoDataFrame or DataFrame"""
        available_columns = GeodataProcessor.get_columns_dataframe(gdf)
        remove_columns = [col for col in columns if col in available_columns]
        if remove_columns:
            return gdf.drop(remove_columns, axis=1)
        return gdf

    @staticmethod
    def get_columns(gdf):
        """Get all columns in GeoDataFrame or DataFrame"""
        return gdf.columns

    @staticmethod
    def geodataframe_bounds(gdf):
        """Return bounding box of GeoDataFrame"""
        return gdf.total_bounds

    def reproject_geodataframe(self, gdf, crs=None):
        """Reproject GeoDataFrame to another CRS"""
        crs = crs if crs else self.crs
        return gdf.to_crs(crs=crs)

    @staticmethod
    def spatial_join(gdf1, gdf2):
        """Perform a spatial join of two GeoDataFrames using intersect"""
        return gpd.sjoin(gdf1, gdf2, how="left", op="intersects")
    
    @staticmethod
    def spatial_join_pol_pts(gdf1, gdf2):
        """Spatial join a point and polygon GeoDataFrame using intersect"""
        joined_poly_pts = gpd.sjoin(gdf1, gdf2, how="inner", predicate="within")
        joined_poly_pts = joined_poly_pts.loc[:,
                                              ~joined_poly_pts.columns.str.endswith(
                                                  ('_left', '_right')
                                              )]
        return joined_poly_pts
    


def pivot_plot(gdf, aggregator, grid_cols=3):
        """
        Plot bar charts for each categorical column in the GeoDataFrame grouped by a specified category.
        """
        if aggregator not in gdf.columns:
            raise ValueError(f"Aggregator column '{aggregator}' not found in GeoDataFrame.")

        categorical_columns = gdf.select_dtypes(include=['object', 'category']).columns.tolist()
        if aggregator in categorical_columns:
            categorical_columns.remove(aggregator)

        if not categorical_columns:
            print("No categorical columns available for plotting.")
            return

        num_plots = len(categorical_columns)
        grid_rows = math.ceil(num_plots / grid_cols)

        fig_height = 5 * grid_rows
        fig_width = 5 * grid_cols
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(fig_width, fig_height))
        
        if num_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, col in enumerate(categorical_columns):
            sns.countplot(data=gdf, x=col, hue=aggregator, ax=axes[i], palette="Set2")
            axes[i].set_title(f'Count of {col} by {aggregator}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)
            if i % grid_cols == 0:
                axes[i].legend(title=aggregator, bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                axes[i].get_legend().remove()

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
        


def plot_categorical_columns(df, drop_columns=None, grid_cols=None, figsize=None):
    """
    Plot bar charts for each categorical column in the DataFrame.

    """
    # Drop specified columns if provided
    if drop_columns:
        df = df.drop(columns=drop_columns, axis=1, errors='ignore')

    # Identify categorical columns
    categorical_columns_names = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not categorical_columns_names:
        print("No categorical columns found in the DataFrame.")
        return
    
    num_plots = len(categorical_columns_names)
    grid_rows = math.ceil(num_plots / grid_cols)

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(figsize[0],
                                                            figsize[1] * grid_rows))
    axes = axes.flatten()   

    for i, col in enumerate(categorical_columns_names):
        value_counts = df[col].value_counts()
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i])
        axes[i].set_title(f'Distribution by {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=90)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_boxplot_by_category(df, category_col, value_col, log_transform=False):
    """
    Plots a boxplot for each category in a DataFrame, with an option for log-scaling the values.

    Parameters:
    df (pd.DataFrame): main dataframe
    category_col (str): col of interest
    value_col (str): numerical aggregator
    log_transform (bool): log transformation

    Returns:
    None: Displays the boxplot.
    """
    transformed_values = df[value_col].copy()

    if log_transform:
        transformed_values = transformed_values.replace(0, np.nan)  
        transformed_values = np.log1p(transformed_values)  
    plt.figure(figsize=(12, 8))
    plt.boxplot(
        [transformed_values[df[category_col] == cat] for cat in df[category_col].unique()],
                labels=df[category_col].unique(), patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                medianprops=dict(color='red'),
                vert=False
        )  

    title = f'Boxplot of {"log-transformed " if log_transform else ""}{value_col} by {category_col}'
    plt.title(title)
    plt.xlabel(f'{"Log-transformed " if log_transform else ""}{value_col}')
    plt.ylabel(category_col)

    plt.show()

def plot_histograms_by_category(df, category_col, value_col, bins=10, rows=4, figsize=(18, 15)):
    """
    Plots histograms for each category, with a defined grid style

    Parameters:
    df (pd.DataFrame): Main data.
    category_col (str): col of interest.
    value_col (str): numerical aggregator.
    bins (int): histogram bins. Default is 10.
    rows (int): Number of grids, default 4
    figsize (tuple): TDefault is (16, 12).

    """
    categories = df[category_col].unique()
    num_categories = len(categories)
    cols = int(np.ceil(num_categories / rows))

    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()
    for i, category in enumerate(categories):
        ax = axes[i]
        sns.histplot(df[df[category_col] == category][value_col], bins=bins, kde=False, ax=ax)
        ax.set_title(f'{category}')
        ax.set_xlabel(value_col)
        ax.set_ylabel('Frequency')
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.show()


def calculate_aggregated_stats(df, group_col, value_col, agg_type='median', percentile=None):
    """
    Calculate aggregated statistics by grouping the DataFrame by a specified column.

    Parameters:
    df (pd.DataFrame): Main dataframe
    group_col (str): col of interest
    value_col (str): aggregating column
    agg_type (str): The type of aggregation ('median' or 'percentile'). Default is 'median'.
    percentile (float): The percentile to agg_type is . Should be between 0 and 100.

    Returns:
    pd.DataFrame: A DataFrame with aggregated statistics and ranks, rounded to 2 decimal places.
    """

    if agg_type == 'percentile' and percentile is not None:
        agg_func = lambda x: x.quantile(percentile / 100)
        agg_type_col = f'{percentile}th Percentile'
    else:
        agg_func = agg_type
        agg_type_col = agg_type.capitalize()

    agg_df = df.groupby(group_col)[value_col].agg(['min', 'max', 'mean', agg_func, 'count', 'std'])
    agg_df.columns = ['min', 'max', 'mean', agg_type_col, 'count', 'std']
    agg_df = agg_df.round(2)
    agg_df['Rank'] = agg_df[agg_type_col].rank(ascending=False)

    agg_df = agg_df.sort_values('Rank')

    return agg_df

def plot_correlation_heatmap(
    data_frame: pd.DataFrame,
    method: str = 'pearson',
    figsize: tuple = (15, 15)
) -> ggplot:
    """
    Computes the correlation matrix for specified columns, renames columns, and returns a half-heatmap plot.
    """

    data_corr = data_frame.corr(method=method)
    data_corr_long = data_corr.stack().reset_index()
    data_corr_long.columns = ["Val1", "Val2", "Correlation"]

    # Create the heatmap plot
    heatmap = (
        ggplot(data_corr_long) +
        geom_tile(aes(x="Val1", y="Val2", fill="Correlation")) +
        geom_text(aes(x="Val1", y="Val2", label="Correlation"), format_string='{:.2f}', size=10) +
        labs(title=f" Correlation Heatmap (Method: {method})") +
        theme(
            figure_size=figsize,
            plot_title=element_text(hjust=0.5, size=20),
            axis_text_x=element_text(angle=90, hjust=1, size=14),
            axis_text_y=element_text(size=14),
            legend_title=element_text(size=18),
            legend_text=element_text(size=15)
        ) +
        scale_fill_cmap(cmap_name="Blues")
    )

    return heatmap


def add_y_offset_to_geometry(gdf, indices, offset=400):
    """
    Adds an offset to the y-coordinate of the specified rows in the GeoDataFrame.
    """
    gdf = gdf.copy()

    for idx in indices:
        point = gdf.loc[idx, 'geometry']
        if isinstance(point, Point):
            new_point = Point(point.x + offset, point.y)
            gdf.at[idx, 'geometry'] = new_point
    return gdf

def print_with_significant_figures(df, sig_figs=5):
    """
    Print a pandas DataFrame with values rounded to a specified number of significant figures.
    """
    def format_value(value):
        if isinstance(value, (int, float)):
            return f"{value:.{sig_figs}g}"
        return value  
