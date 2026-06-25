"""
SALES DATA ANALYSIS DASHBOARD - COMPLETE PROJECT
Author: Data Analytics Internship (Navyan)
Purpose: Load, clean, analyze, and visualize sales data
Requirements: superstore.csv from Kaggle

WORKFLOW:
1. Load & validate data
2. Clean & prepare data
3. Exploratory data analysis
4. Calculate key metrics
5. Create static visualizations
6. Create interactive dashboard
7. Export cleaned data for Power BI/Tableau
8. Generate comprehensive report
"""

# ============================================
# SECTION 1: IMPORTS & CONFIGURATION
# ============================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set visualization styles
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Constants
REQUIRED_COLUMNS = ['Order ID', 'Order Date', 'Sales', 'Profit', 'Category', 
                    'Region', 'Segment', 'Product Name', 'Quantity', 'Discount']
MONTH_ORDER = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
OUTPUT_DIR = './output'
SAMPLE_SIZE = 100  # For head/sample displays


# ============================================
# SECTION 2: UTILITY FUNCTIONS
# ============================================

def setup_output_directory():
    """Create output directory for exports"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"✅ Created output directory: {OUTPUT_DIR}")


def validate_dataframe(df, required_cols=None):
    """
    Validate that DataFrame has required columns
    
    Args:
        df: pandas DataFrame
        required_cols: list of column names to check
    
    Returns:
        bool: True if valid, False otherwise
    """
    if df is None or df.empty:
        print("❌ ERROR: DataFrame is empty or None")
        return False
    
    if required_cols is None:
        required_cols = REQUIRED_COLUMNS
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ ERROR: Missing required columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        return False
    
    return True


# ============================================
# SECTION 3: DATA LOADING
# ============================================

def load_data(filepath='superstore.csv'):
    """
    Load sales dataset from CSV
    
    Args:
        filepath: path to CSV file (default: superstore.csv)
    
    Returns:
        pandas DataFrame or None if load fails
    """
    print("\n" + "="*70)
    print("📥 LOADING DATA")
    print("="*70)
    
    # Check file exists
    if not os.path.exists(filepath):
        print(f"\n❌ ERROR: File not found: {filepath}")
        print(f"\n📋 To download data:")
        print("   1. Visit: https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting")
        print("   2. Download 'superstore.csv'")
        print(f"   3. Place it in the working directory: {os.getcwd()}")
        return None
    
    try:
        df = pd.read_csv(filepath)
        print(f"\n✅ Successfully loaded: {filepath}")
        print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"   Date range: {df['Order Date'].min()} to {df['Order Date'].max()}")
        return df
    
    except Exception as e:
        print(f"❌ ERROR loading file: {str(e)}")
        return None


# ============================================
# SECTION 4: DATA EXPLORATION
# ============================================

def explore_data(df):
    """
    Explore dataset structure and content
    
    Args:
        df: pandas DataFrame
    """
    if not validate_dataframe(df):
        return
    
    print("\n" + "="*70)
    print("🔍 DATA EXPLORATION")
    print("="*70)
    
    print(f"\n📌 Dataset Dimensions:")
    print(f"   Rows: {df.shape[0]:,}")
    print(f"   Columns: {df.shape[1]}")
    
    print(f"\n📌 Column Names & Types:")
    for col in df.columns:
        print(f"   • {col:20} → {df[col].dtype}")
    
    print(f"\n📌 First 5 Rows:")
    print(df.head().to_string())
    
    print(f"\n📌 Data Types Summary:")
    print(df.dtypes)
    
    print(f"\n📌 Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✅ No missing values found")
    else:
        print(missing[missing > 0])
    
    print(f"\n📌 Duplicate Rows: {df.duplicated().sum()}")
    
    print(f"\n📌 Numeric Columns Statistics:")
    print(df.describe().round(2))


# ============================================
# SECTION 5: DATA CLEANING
# ============================================

def clean_data(df):
    """
    Clean and prepare data for analysis
    
    Args:
        df: pandas DataFrame (raw)
    
    Returns:
        pandas DataFrame (cleaned)
    """
    if not validate_dataframe(df):
        return None
    
    print("\n" + "="*70)
    print("🧹 DATA CLEANING")
    print("="*70)
    
    df_clean = df.copy()
    
    # Step 1: Handle Missing Values
    print(f"\n1️⃣ Handling Missing Values")
    missing_before = df_clean.isnull().sum().sum()
    
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if col in ['Sales', 'Profit', 'Quantity', 'Discount']:
                # Numeric columns: fill with 0
                df_clean[col] = df_clean[col].fillna(0)
            else:
                # Categorical columns: fill with 'Unknown'
                df_clean[col] = df_clean[col].fillna('Unknown')
    
    missing_after = df_clean.isnull().sum().sum()
    print(f"   Status: {missing_before} → {missing_after} missing values")
    if missing_before == 0:
        print(f"   ✅ No missing values to handle")
    
    # Step 2: Remove Duplicates
    print(f"\n2️⃣ Removing Duplicate Rows")
    duplicates_before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = duplicates_before - len(df_clean)
    print(f"   ✅ Removed {duplicates_removed} duplicate rows")
    
    # Step 3: Fix Date Formats
    print(f"\n3️⃣ Converting Date Columns")
    try:
        df_clean['Order Date'] = pd.to_datetime(df_clean['Order Date'])
        if 'Ship Date' in df_clean.columns:
            df_clean['Ship Date'] = pd.to_datetime(df_clean['Ship Date'])
        print(f"   ✅ Date columns converted to datetime")
    except Exception as e:
        print(f"   ⚠️ Warning: Could not convert dates: {str(e)}")
    
    # Step 4: Create Derived Columns
    print(f"\n4️⃣ Creating Derived Columns")
    
    # Time-based columns
    df_clean['Year'] = df_clean['Order Date'].dt.year
    df_clean['Month'] = df_clean['Order Date'].dt.month
    df_clean['Month Name'] = df_clean['Order Date'].dt.month_name()
    df_clean['Quarter'] = df_clean['Order Date'].dt.quarter
    df_clean['Year-Month'] = df_clean['Order Date'].dt.strftime('%Y-%m')
    df_clean['Day of Week'] = df_clean['Order Date'].dt.day_name()
    
    # Financial metrics
    df_clean['Profit Margin %'] = np.where(
        df_clean['Sales'] > 0,
        (df_clean['Profit'] / df_clean['Sales']) * 100,
        0  # Treat zero sales as 0% margin
    )
    
    # Replace inf values with 0
    df_clean['Profit Margin %'] = df_clean['Profit Margin %'].replace(
        [np.inf, -np.inf], 0
    )
    
    print(f"   ✅ Added 7 derived columns:")
    print(f"      - Year, Month, Month Name, Quarter, Year-Month")
    print(f"      - Day of Week, Profit Margin %")
    
    # Step 5: Data Validation
    print(f"\n5️⃣ Data Validation")
    print(f"   • Negative sales: {(df_clean['Sales'] < 0).sum()}")
    print(f"   • Negative profit: {(df_clean['Profit'] < 0).sum()}")
    print(f"   • Valid records: {len(df_clean):,}")
    
    print(f"\n📊 Cleaned Dataset Summary:")
    print(f"   Shape: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
    print(f"   Date range: {df_clean['Order Date'].min().date()} to {df_clean['Order Date'].max().date()}")
    print(f"   ✅ Data cleaning completed successfully")
    
    return df_clean


# ============================================
# SECTION 6: KEY METRICS CALCULATION
# ============================================

def calculate_metrics(df):
    """
    Calculate key business metrics
    
    Args:
        df: pandas DataFrame (cleaned)
    
    Returns:
        dict: metrics dictionary
    """
    if not validate_dataframe(df):
        return None
    
    print("\n" + "="*70)
    print("📊 KEY METRICS ANALYSIS")
    print("="*70)
    
    # Calculate metrics
    total_sales = df['Sales'].sum()
    total_profit = df['Profit'].sum()
    total_orders = df['Order ID'].nunique()
    
    metrics = {
        'Total Sales ($)': total_sales,
        'Total Profit ($)': total_profit,
        'Total Orders': total_orders,
        'Total Quantity': df['Quantity'].sum(),
        'Average Order Value ($)': total_sales / total_orders if total_orders > 0 else 0,
        'Average Profit per Order ($)': total_profit / total_orders if total_orders > 0 else 0,
        'Overall Profit Margin (%)': (total_profit / total_sales * 100) if total_sales > 0 else 0,
        'Unique Customers': df['Customer ID'].nunique() if 'Customer ID' in df.columns else 'N/A',
        'Unique Products': df['Product Name'].nunique() if 'Product Name' in df.columns else 'N/A',
        'Unique Regions': df['Region'].nunique(),
        'Unique Categories': df['Category'].nunique(),
    }
    
    # Display metrics
    print("\n")
    for metric_name, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            print(f"   {metric_name:.<40} {value:>15,}")
        elif isinstance(value, (float, np.floating)):
            print(f"   {metric_name:.<40} {value:>15,.2f}")
        else:
            print(f"   {metric_name:.<40} {value:>15}")
    
    return metrics


# ============================================
# SECTION 7: CATEGORY & PRODUCT ANALYSIS
# ============================================

def analyze_categories(df):
    """
    Analyze performance by category and sub-category
    
    Args:
        df: pandas DataFrame (cleaned)
    """
    if not validate_dataframe(df, ['Category', 'Sales', 'Profit']):
        return
    
    print("\n" + "="*70)
    print("📦 CATEGORY ANALYSIS")
    print("="*70)
    
    # Category performance
    category_perf = df.groupby('Category').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'count',
        'Quantity': 'sum'
    }).round(2)
    category_perf.columns = ['Total Sales', 'Total Profit', 'Order Count', 'Total Qty']
    category_perf['Profit Margin %'] = (
        category_perf['Total Profit'] / category_perf['Total Sales'] * 100
    ).round(2)
    category_perf = category_perf.sort_values('Total Sales', ascending=False)
    
    print(f"\n🏆 Category Performance:")
    print(category_perf.to_string())
    
    # Top products
    if 'Product Name' in df.columns:
        print(f"\n📈 Top 10 Products by Sales:")
        top_products = df.groupby('Product Name')['Sales'].sum().sort_values(
            ascending=False
        ).head(10)
        for idx, (product, sales) in enumerate(top_products.items(), 1):
            print(f"   {idx:2}. {product:40} ${sales:>12,.2f}")


# ============================================
# SECTION 8: REGIONAL ANALYSIS
# ============================================

def analyze_regions(df):
    """
    Analyze performance by region and state
    
    Args:
        df: pandas DataFrame (cleaned)
    """
    if not validate_dataframe(df, ['Region', 'Sales', 'Profit']):
        return
    
    print("\n" + "="*70)
    print("🌍 REGIONAL ANALYSIS")
    print("="*70)
    
    # Regional performance
    regional_perf = df.groupby('Region').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'count'
    }).round(2)
    regional_perf.columns = ['Total Sales', 'Total Profit', 'Order Count']
    regional_perf['Profit Margin %'] = (
        regional_perf['Total Profit'] / regional_perf['Total Sales'] * 100
    ).round(2)
    regional_perf['Avg Order Value'] = (
        regional_perf['Total Sales'] / regional_perf['Order Count']
    ).round(2)
    regional_perf = regional_perf.sort_values('Total Sales', ascending=False)
    
    print(f"\n📊 Regional Performance:")
    print(regional_perf.to_string())


# ============================================
# SECTION 9: TIME TRENDS ANALYSIS
# ============================================

def analyze_trends(df):
    """
    Analyze sales trends over time
    
    Args:
        df: pandas DataFrame (cleaned)
    """
    if not validate_dataframe(df, ['Order Date', 'Sales', 'Profit']):
        return
    
    print("\n" + "="*70)
    print("📅 TIME TRENDS ANALYSIS")
    print("="*70)
    
    # Monthly trends
    monthly = df.groupby('Year-Month').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'count'
    }).round(2)
    monthly.columns = ['Monthly Sales', 'Monthly Profit', 'Order Count']
    monthly.index = pd.to_datetime(monthly.index)
    monthly = monthly.sort_index()
    
    print(f"\n📈 Monthly Trends (Last 6 months):")
    print(monthly.tail(6).to_string())
    
    # Quarterly trends
    quarterly = df.groupby('Quarter').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).round(2)
    quarterly.columns = ['Total Sales', 'Total Profit']
    quarterly['Profit Margin %'] = (
        quarterly['Total Profit'] / quarterly['Total Sales'] * 100
    ).round(2)
    
    print(f"\n📊 Quarterly Performance:")
    print(quarterly.to_string())
    
    # Growth calculation
    if len(monthly) > 1:
        first_month = monthly['Monthly Sales'].iloc[0]
        last_month = monthly['Monthly Sales'].iloc[-1]
        total_growth = ((last_month - first_month) / first_month * 100) if first_month > 0 else 0
        print(f"\n📊 Overall Growth: {total_growth:.2f}%")


# ============================================
# SECTION 10: STATIC VISUALIZATIONS
# ============================================

def create_static_visualizations(df):
    """
    Create static charts using Matplotlib/Seaborn
    
    Args:
        df: pandas DataFrame (cleaned)
    """
    if not validate_dataframe(df):
        return
    
    print("\n" + "="*70)
    print("📊 CREATING STATIC VISUALIZATIONS")
    print("="*70)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sales Dashboard - Static Visualizations', fontsize=16, fontweight='bold')
    
    # Chart 1: Sales by Category (Bar)
    try:
        category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
        colors_cat = plt.cm.Set3(np.linspace(0, 1, len(category_sales)))
        
        bars = axes[0, 0].bar(category_sales.index, category_sales.values, color=colors_cat)
        axes[0, 0].set_title('Sales by Category', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Category')
        axes[0, 0].set_ylabel('Sales ($)')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                          f'${height:,.0f}',
                          ha='center', va='bottom', fontsize=9)
        print("   ✅ Chart 1: Sales by Category")
    except Exception as e:
        print(f"   ⚠️ Chart 1 failed: {str(e)}")
    
    # Chart 2: Sales by Region (Bar)
    try:
        region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
        colors_region = plt.cm.Set2(np.linspace(0, 1, len(region_sales)))
        
        bars = axes[0, 1].bar(region_sales.index, region_sales.values, color=colors_region)
        axes[0, 1].set_title('Sales by Region', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Region')
        axes[0, 1].set_ylabel('Sales ($)')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'${height:,.0f}',
                          ha='center', va='bottom', fontsize=9)
        print("   ✅ Chart 2: Sales by Region")
    except Exception as e:
        print(f"   ⚠️ Chart 2 failed: {str(e)}")
    
    # Chart 3: Monthly Sales Trend (Line)
    try:
        monthly_sales = df.groupby('Month Name')['Sales'].sum()
        monthly_sales = monthly_sales.reindex(MONTH_ORDER)
        
        axes[1, 0].plot(range(len(monthly_sales)), monthly_sales.values, 
                       marker='o', linewidth=2, markersize=8, color='#FF6B6B')
        axes[1, 0].fill_between(range(len(monthly_sales)), monthly_sales.values, 
                               alpha=0.3, color='#FF6B6B')
        axes[1, 0].set_title('Monthly Sales Trend', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Sales ($)')
        axes[1, 0].set_xticks(range(len(MONTH_ORDER)))
        axes[1, 0].set_xticklabels([m[:3] for m in MONTH_ORDER], rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        print("   ✅ Chart 3: Monthly Sales Trend")
    except Exception as e:
        print(f"   ⚠️ Chart 3 failed: {str(e)}")
    
    # Chart 4: Profit Distribution (Pie)
    try:
        profit_by_cat = df.groupby('Category')['Profit'].sum()
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(profit_by_cat)))
        explode = [0.05] * len(profit_by_cat)  # Dynamic explode
        
        wedges, texts, autotexts = axes[1, 1].pie(
            profit_by_cat.values,
            labels=profit_by_cat.index,
            autopct='%1.1f%%',
            colors=colors_pie,
            explode=explode,
            startangle=90
        )
        axes[1, 1].set_title('Profit Distribution by Category', fontsize=12, fontweight='bold')
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        print("   ✅ Chart 4: Profit Distribution")
    except Exception as e:
        print(f"   ⚠️ Chart 4 failed: {str(e)}")
    
    plt.tight_layout()
    
    # Save figure
    filepath = f"{OUTPUT_DIR}/01_sales_dashboard_static.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {filepath}")


# ============================================
# SECTION 11: INTERACTIVE DASHBOARD
# ============================================

def create_interactive_dashboard(df):
    """
    Create interactive Plotly dashboard
    
    Args:
        df: pandas DataFrame (cleaned)
    """
    if not validate_dataframe(df):
        return
    
    print("\n" + "="*70)
    print("🎨 CREATING INTERACTIVE DASHBOARD")
    print("="*70)
    
    try:
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Sales by Category', 'Sales by Region', 'Top 5 Products',
                'Profit by Category', 'Sales Trend', 'Segment Distribution',
                'Profit Margin by Region', 'Sales vs Profit', 'Monthly Breakdown'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'pie'}, {'type': 'scatter'}, {'type': 'pie'}],
                [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'bar'}]
            ]
        )
        
        # Chart 1: Sales by Category
        cat_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=cat_sales.index, y=cat_sales.values, name='Sales',
                   marker_color='lightblue', text=[f'${x:,.0f}' for x in cat_sales.values],
                   textposition='outside'),
            row=1, col=1
        )
        
        # Chart 2: Sales by Region
        reg_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=reg_sales.index, y=reg_sales.values, name='Sales',
                   marker_color='lightcoral', text=[f'${x:,.0f}' for x in reg_sales.values],
                   textposition='outside'),
            row=1, col=2
        )
        
        # Chart 3: Top 5 Products
        if 'Product Name' in df.columns:
            top5 = df.groupby('Product Name')['Sales'].sum().nlargest(5)
            fig.add_trace(
                go.Bar(x=top5.values, y=top5.index, orientation='h', name='Sales',
                       marker_color='lightgreen', text=[f'${x:,.0f}' for x in top5.values],
                       textposition='outside'),
                row=1, col=3
            )
        
        # Chart 4: Profit by Category (Pie)
        cat_profit = df.groupby('Category')['Profit'].sum()
        fig.add_trace(
            go.Pie(labels=cat_profit.index, values=cat_profit.values,
                   name='Profit', textinfo='label+percent'),
            row=2, col=1
        )
        
        # Chart 5: Sales Trend (Line)
        monthly = df.groupby('Month Name')['Sales'].sum().reindex(MONTH_ORDER)
        fig.add_trace(
            go.Scatter(x=MONTH_ORDER, y=monthly.values, mode='lines+markers',
                      name='Sales', line=dict(color='blue', width=2),
                      fill='tozeroy'),
            row=2, col=2
        )
        
        # Chart 6: Segment Distribution (Pie)
        if 'Segment' in df.columns:
            seg_sales = df.groupby('Segment')['Sales'].sum()
            fig.add_trace(
                go.Pie(labels=seg_sales.index, values=seg_sales.values,
                       name='Segment', textinfo='label+percent'),
                row=2, col=3
            )
        
        # Chart 7: Profit Margin by Region (Bar)
        reg_margin = df.groupby('Region').agg({
            'Profit': 'sum',
            'Sales': 'sum'
        })
        reg_margin['Margin %'] = (reg_margin['Profit'] / reg_margin['Sales'] * 100).round(2)
        reg_margin = reg_margin.sort_values('Margin %', ascending=False)
        
        fig.add_trace(
            go.Bar(x=reg_margin.index, y=reg_margin['Margin %'], name='Profit Margin %',
                   marker_color=['green' if x > 0 else 'red' for x in reg_margin['Margin %']],
                   text=[f'{x:.1f}%' for x in reg_margin['Margin %']],
                   textposition='outside'),
            row=3, col=1
        )
        
        # Chart 8: Sales vs Profit (Scatter)
        if 'Product Name' in df.columns:
            product_data = df.groupby('Product Name').agg({
                'Sales': 'sum',
                'Profit': 'sum'
            }).head(20)
            fig.add_trace(
                go.Scatter(x=product_data['Sales'], y=product_data['Profit'],
                          mode='markers', name='Products',
                          marker=dict(size=10, color=product_data['Profit'],
                                     colorscale='Viridis', showscale=False)),
                row=3, col=2
            )
        
        # Chart 9: Monthly Breakdown (Bar)
        monthly_profit = df.groupby('Month Name')['Profit'].sum().reindex(MONTH_ORDER)
        fig.add_trace(
            go.Bar(x=MONTH_ORDER, y=monthly_profit.values, name='Monthly Profit',
                   marker_color='orange', text=[f'${x:,.0f}' for x in monthly_profit.values],
                   textposition='outside'),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=1400,
            width=1800,
            showlegend=False,
            title_text="<b>Sales Analysis Dashboard</b>",
            title_font_size=20,
            template='plotly_white'
        )
        
        # Save dashboard
        filepath = f"{OUTPUT_DIR}/02_interactive_dashboard.html"
        fig.write_html(filepath)
        print(f"✅ Saved interactive dashboard: {filepath}")
        
    except Exception as e:
        print(f"❌ Error creating interactive dashboard: {str(e)}")


# ============================================
# SECTION 12: EXPORT DATA
# ============================================

def export_data(df, metrics):
    """
    Export cleaned data and metrics for Power BI/Tableau
    
    Args:
        df: pandas DataFrame (cleaned)
        metrics: dict of calculated metrics
    """
    if not validate_dataframe(df):
        return
    
    print("\n" + "="*70)
    print("💾 EXPORTING DATA")
    print("="*70)
    
    try:
        # Export cleaned data
        filepath_csv = f"{OUTPUT_DIR}/superstore_cleaned.csv"
        df.to_csv(filepath_csv, index=False)
        print(f"\n✅ Cleaned data: {filepath_csv}")
        print(f"   Rows: {len(df):,} | Columns: {len(df.columns)}")
        
        # Export summary statistics
        filepath_stats = f"{OUTPUT_DIR}/data_summary_statistics.csv"
        df.describe().to_csv(filepath_stats)
        print(f"✅ Summary statistics: {filepath_stats}")
        
        # Export metrics
        filepath_metrics = f"{OUTPUT_DIR}/key_metrics.csv"
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        metrics_df.to_csv(filepath_metrics, index=False)
        print(f"✅ Key metrics: {filepath_metrics}")
        
        # Export category analysis
        filepath_cat = f"{OUTPUT_DIR}/category_analysis.csv"
        cat_analysis = df.groupby('Category').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'count',
            'Quantity': 'sum'
        }).round(2)
        cat_analysis['Profit Margin %'] = (
            cat_analysis['Profit'] / cat_analysis['Sales'] * 100
        ).round(2)
        cat_analysis.to_csv(filepath_cat)
        print(f"✅ Category analysis: {filepath_cat}")
        
        # Export regional analysis
        filepath_reg = f"{OUTPUT_DIR}/regional_analysis.csv"
        reg_analysis = df.groupby('Region').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'count'
        }).round(2)
        reg_analysis['Profit Margin %'] = (
            reg_analysis['Profit'] / reg_analysis['Sales'] * 100
        ).round(2)
        reg_analysis.to_csv(filepath_reg)
        print(f"✅ Regional analysis: {filepath_reg}")
        
        print(f"\n📁 All exports saved to: {OUTPUT_DIR}/")
        
    except Exception as e:
        print(f"❌ Error during export: {str(e)}")


# ============================================
# SECTION 13: GENERATE REPORT
# ============================================

def generate_report(df, metrics):
    """
    Generate comprehensive analysis report
    
    Args:
        df: pandas DataFrame (cleaned)
        metrics: dict of calculated metrics
    """
    if not validate_dataframe(df):
        return
    
    print("\n" + "="*70)
    print("📝 GENERATING ANALYSIS REPORT")
    print("="*70)
    
    try:
        # Prepare report data
        report_date = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Top products
        top_products = df.groupby('Product Name')['Sales'].sum().nlargest(5) if 'Product Name' in df.columns else None
        
        # Regional breakdown
        regional = df.groupby('Region')['Sales'].sum().to_dict()
        
        # Generate report
        report = f"""
═══════════════════════════════════════════════════════════════════════════
                    SALES DATA ANALYSIS REPORT
═══════════════════════════════════════════════════════════════════════════

Generated: {report_date}
Dataset: {len(df):,} sales records
Date Range: {df['Order Date'].min().date()} to {df['Order Date'].max().date()}

───────────────────────────────────────────────────────────────────────────
EXECUTIVE SUMMARY
───────────────────────────────────────────────────────────────────────────

Total Sales              ${metrics.get('Total Sales ($)', 0):>20,.2f}
Total Profit            ${metrics.get('Total Profit ($)', 0):>20,.2f}
Overall Profit Margin   {metrics.get('Overall Profit Margin (%)', 0):>19,.2f}%

Total Orders            {metrics.get('Total Orders', 0):>20,.0f}
Average Order Value     ${metrics.get('Average Order Value ($)', 0):>19,.2f}
Unique Customers        {metrics.get('Unique Customers', 'N/A'):>20}
Unique Products         {metrics.get('Unique Products', 'N/A'):>20}

───────────────────────────────────────────────────────────────────────────
CATEGORY PERFORMANCE
───────────────────────────────────────────────────────────────────────────
"""
        
        cat_perf = df.groupby('Category').agg({
            'Sales': 'sum',
            'Profit': 'sum'
        }).sort_values('Sales', ascending=False)
        
        for category, row in cat_perf.iterrows():
            margin = (row['Profit'] / row['Sales'] * 100) if row['Sales'] > 0 else 0
            report += f"\n{category:15} | Sales: ${row['Sales']:>12,.0f} | Profit: ${row['Profit']:>10,.0f} | Margin: {margin:>6.2f}%"
        
        # Regional breakdown
        report += f"\n\n───────────────────────────────────────────────────────────────────────────\nREGIONAL BREAKDOWN\n───────────────────────────────────────────────────────────────────────────\n"
        
        for region, sales in regional.items():
            report += f"\n{region:15} | Sales: ${sales:>12,.0f}"
        
        # Top products
        if top_products is not None:
            report += f"\n\n───────────────────────────────────────────────────────────────────────────\nTOP 5 PRODUCTS BY SALES\n───────────────────────────────────────────────────────────────────────────\n"
            for i, (product, sales) in enumerate(top_products.items(), 1):
                report += f"\n{i}. {product[:40]:40} ${sales:>12,.0f}"
        
        report += f"""

───────────────────────────────────────────────────────────────────────────
KEY INSIGHTS & RECOMMENDATIONS
───────────────────────────────────────────────────────────────────────────

1. SALES PERFORMANCE
   • Highest performing category: {df.groupby('Category')['Sales'].sum().idxmax()}
   • Highest performing region: {df.groupby('Region')['Sales'].sum().idxmax()}
   • Monthly average sales: ${df.groupby('Year-Month')['Sales'].sum().mean():,.2f}

2. PROFITABILITY ANALYSIS
   • Categories with positive profit: {len(df[df['Profit'] > 0]):,} transactions
   • Categories with negative profit: {len(df[df['Profit'] < 0]):,} transactions
   • Average profit per transaction: ${df['Profit'].mean():,.2f}

3. OPERATIONAL INSIGHTS
   • Total transactions analyzed: {len(df):,}
   • Average transaction value: ${df['Sales'].mean():,.2f}
   • Standard deviation in profit: ${df['Profit'].std():,.2f}

4. RECOMMENDATIONS
   ✓ Focus marketing budget on {df.groupby('Category')['Sales'].sum().idxmax()} category
   ✓ Expand operations in {df.groupby('Region')['Sales'].sum().idxmax()} region
   ✓ Investigate {len(df[df['Profit'] < 0]):,} low-profit transactions
   ✓ Maintain quality standards in top-performing products
   ✓ Consider inventory optimization for seasonal trends

───────────────────────────────────────────────────────────────────────────
NEXT STEPS
───────────────────────────────────────────────────────────────────────────

1. Import cleaned data (superstore_cleaned.csv) into Power BI or Tableau
2. Create interactive dashboards with filters by Region, Category, Date
3. Set up automated reporting for monthly performance tracking
4. Develop predictive models for sales forecasting

═══════════════════════════════════════════════════════════════════════════
"""
        
        # Save report
        filepath = f"{OUTPUT_DIR}/analysis_report.txt"
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"\n✅ Report saved: {filepath}")
        
    except Exception as e:
        print(f"❌ Error generating report: {str(e)}")


# ============================================
# SECTION 14: MAIN EXECUTION
# ============================================

def main():
    """
    Main execution function - orchestrates the entire analysis workflow
    """
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "SALES DATA ANALYSIS DASHBOARD - COMPLETE PROJECT".center(68) + "║")
    print("║" + "Navyan Data Analytics Internship".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    # Setup
    setup_output_directory()
    
    # Step 1: Load Data
    df = load_data('superstore.csv')
    if df is None:
        print("\n❌ FATAL ERROR: Could not load data. Exiting.")
        sys.exit(1)
    
    # Step 2: Explore Data
    explore_data(df)
    
    # Step 3: Clean Data
    df_clean = clean_data(df)
    if df_clean is None:
        print("\n❌ FATAL ERROR: Data cleaning failed. Exiting.")
        sys.exit(1)
    
    # Step 4: Calculate Metrics
    metrics = calculate_metrics(df_clean)
    if metrics is None:
        print("\n❌ ERROR: Could not calculate metrics.")
    
    # Step 5: Analyze Categories
    analyze_categories(df_clean)
    
    # Step 6: Analyze Regions
    analyze_regions(df_clean)
    
    # Step 7: Analyze Trends
    analyze_trends(df_clean)
    
    # Step 8: Create Static Visualizations
    create_static_visualizations(df_clean)
    
    # Step 9: Create Interactive Dashboard
    create_interactive_dashboard(df_clean)
    
    # Step 10: Export Data
    export_data(df_clean, metrics)
    
    # Step 11: Generate Report
    generate_report(df_clean, metrics)
    
    # Final Summary
    print("\n" + "="*70)
    print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📁 Output Directory: {OUTPUT_DIR}/")
    print("\n📊 Generated Files:")
    print("   1. 01_sales_dashboard_static.png - Static visualizations")
    print("   2. 02_interactive_dashboard.html - Interactive Plotly dashboard")
    print("   3. superstore_cleaned.csv - Cleaned data for Power BI/Tableau")
    print("   4. data_summary_statistics.csv - Statistical summary")
    print("   5. key_metrics.csv - Key business metrics")
    print("   6. category_analysis.csv - Category breakdown")
    print("   7. regional_analysis.csv - Regional breakdown")
    print("   8. analysis_report.txt - Comprehensive analysis report")
    print("\n📈 Next Steps:")
    print("   • Open 02_interactive_dashboard.html in your browser")
    print("   • Import superstore_cleaned.csv into Power BI or Tableau")
    print("   • Create filters for Region, Category, and Date Range")
    print("\n✅ All analysis complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
