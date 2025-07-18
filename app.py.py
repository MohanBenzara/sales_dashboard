# ==============================================================================
# BUSINESS INTELLIGENCE HUB (VERSION 58.0 - FINAL CHART & UI REFINEMENTS)
# ==============================================================================
#
# INSTRUCTIONS:
# 1. Make sure you have the required libraries installed:
#    pip install pandas dash plotly dash-bootstrap-components openpyxl prophet
#
# 2. This version includes the final refinements to charts and profitability logic.
#
# ==============================================================================

import base64
import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from datetime import datetime
from prophet import Prophet
from plotly.subplots import make_subplots

# --- 1. APP INITIALIZATION ---
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, dbc.icons.FONT_AWESOME], suppress_callback_exceptions=True)

# Add this line for the deployment server
server = app.server

# --- 2. APP LAYOUT ---
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Img(src="https://benzara.com/cdn/shop/files/download.png?v=1690743366&width=600", height="60px", className="me-3"),
                html.Div([
                    html.H1("Sales & Product Analysis Hub", className="text-primary mb-0"),
                    html.P("Developed by Import Team", style={'fontSize': 'medium', 'color': 'grey', 'fontStyle': 'italic', 'marginTop': '0px'})
                ], className="text-center")
            ], className="d-flex align-items-center")
        ], width="auto")
    ], justify="center", align="center", className="my-4"),

    # Data Stores
    dcc.Store(id='sales-data-store'),
    dcc.Store(id='reorder-data-store'),
    dcc.Store(id='margins-data-store'),
    dcc.Store(id='import-list-store'),
    dcc.Store(id='merged-data-store'),
    dcc.Store(id='abc-analysis-data-store'),
    dcc.Download(id="download-csv"),
    dcc.Download(id="download-abc-component"),

    # Upload Section
    dbc.Row([dbc.Col(dbc.Button("Show/Hide Upload Section", id="collapse-upload-button", className="mb-2"), width=12)]),
    dbc.Collapse(
        dbc.Row([
            dbc.Col(dcc.Upload(id={'type': 'upload', 'index': 'sales'}, children=html.Div(['1. Upload Sales Data']), className='upload-box'), width=3),
            dbc.Col(dcc.Upload(id={'type': 'upload', 'index': 'reorder'}, children=html.Div(['2. Upload Reorder Data']), className='upload-box'), width=3),
            dbc.Col(dcc.Upload(id={'type': 'upload', 'index': 'margins'}, children=html.Div(['3. Upload Margins & Costs']), className='upload-box'), width=3),
            dbc.Col(dcc.Upload(id={'type': 'upload', 'index': 'import'}, children=html.Div(['4. Upload Import Item List']), className='upload-box'), width=3),
        ], className="mb-4"),
        id="upload-collapse", is_open=True
    ),
    html.Div(id='processing-status', className='text-center text-muted mb-4'),

    # --- Tabbed Layout ---
    dbc.Tabs(id="dashboard-tabs", active_tab="tab-overview", children=[
        dbc.Tab(label="Sales Overview", tab_id="tab-overview", children=[
            html.Div(id='main-dashboard-content', style={'display': 'none'}, children=[
                dbc.Card(dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dcc.DatePickerRange(id='date-range-picker', className="w-100"), width=12, lg=3),
                        dbc.Col(dcc.Dropdown(id='item-dropdown', multi=True, placeholder="Filter by Item..."), width=12, lg=3),
                        dbc.Col(dcc.Dropdown(id='customer-dropdown', multi=True, placeholder="Filter by Customer..."), width=12, lg=3),
                        dbc.Col(dcc.Dropdown(id='category-dropdown', multi=True, placeholder="Filter by Category..."), width=12, lg=3, id='category-filter-col', style={'display': 'none'}),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col(dbc.Switch(id='import-only-switch', label="Show Import Items Only", value=False), width="auto"),
                        dbc.Col(dbc.Button("Clear All Filters", id="clear-filters-button", color="danger", outline=True), width="auto", className="ms-auto")
                    ])
                ])),
                html.Div(id='deep-dive-section', className="mt-4"),
                dbc.Row(id='kpi-cards-row', className="my-4"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='monthly-quantity-chart'), width=12, lg=6),
                    dbc.Col(dcc.Graph(id='customer-sales-chart'), width=12, lg=6)
                ], className="mb-4"),
                html.Div(id='price-trend-div', style={'display': 'none'}, children=[
                    dbc.Row([dbc.Col(dcc.Graph(id='price-trend-chart'), width=12)])
                ]),
                html.Div(id='import-charts-section', style={'display': 'none'}, children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='category-sales-pie-chart'), width=12, lg=6),
                        dbc.Col(dcc.Graph(id='top-items-bar-chart'), width=12, lg=6)
                    ])
                ]),
                dbc.Row([dbc.Col(html.Div([
                    html.H4("Transaction Details"),
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(id='table-month-filter', placeholder="Filter Table by Month..."), width=12, lg=4, className="mb-2"),
                        dbc.Col(dbc.Button("Export Table to CSV", id="export-csv-button", color="success"), width="auto", className="mb-2")
                    ]),
                    dash_table.DataTable(id='main-data-table', page_size=10, sort_action="native")
                ]))])
            ])
        ]),

        dbc.Tab(label="Inventory Analysis", tab_id="tab-inventory", children=[
            dbc.Card(dbc.CardBody([
                dbc.Row([dbc.Col(dbc.Switch(id='inventory-import-only-toggle', label="Show Import Items Only", value=True), width='auto')], className="mb-3"),
                html.H4("Inventory Value & Health", className="mb-3"),
                dbc.Row(id='inventory-kpi-row', className="my-4 g-3"),
                html.Hr(),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='in-stock-percentage-chart'), width=12, lg=6),
                    dbc.Col([
                        html.H5("ABC Analysis"),
                        html.P("Class A: Top 80% of consumption value. Class B: Next 15%. Class C: Bottom 5%.", className="text-muted small"),
                        html.Div(id='abc-summary-table'),
                        dbc.Button("Download ABC Analysis Details", id="download-abc-csv", color="secondary", outline=True, className="mt-2 w-100")
                    ], width=12, lg=6),
                ], className="my-4"),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.H4("At-Risk Inventory Alert"),
                        html.P("Items with less than 60 days of total supply.", className="text-muted"),
                        dash_table.DataTable(id='at-risk-table', page_size=5, sort_action="native")
                    ]),
                ], className="my-4")
            ]))
        ]),
        
        dbc.Tab(label="Profitability Analysis", tab_id="tab-profit", children=[
            dbc.Card(dbc.CardBody([
                html.H4("Profitability Overview"),
                dbc.Row([
                    dbc.Col(dcc.Dropdown(id='profit-item-dropdown', multi=True, placeholder="Filter by Item...")),
                    dbc.Col(dcc.Dropdown(id='profit-customer-dropdown', multi=True, placeholder="Filter by Customer...")),
                    dbc.Col(dcc.Dropdown(id='profit-category-dropdown', multi=True, placeholder="Filter by Category...")),
                ], className="mb-3"),
                dbc.Row(id='profit-kpi-row', className="my-4 g-3"),
                html.Hr(),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='profit-by-customer-chart'), width=12, lg=6),
                    dbc.Col(dcc.Graph(id='top-profitable-items-chart'), width=12, lg=6)
                ], className="my-4"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='margin-by-category-chart'), width=12)
                ])
            ]))
        ]),

        dbc.Tab(label="Forecasting", tab_id="tab-forecast", children=[
             dbc.Card(dbc.CardBody([
                html.H4("Sales & Demand Forecasting"),
                dbc.Row([
                    dbc.Col(dcc.Dropdown(id='forecast-item-dropdown', placeholder="Select an item to forecast..."), width=12, lg=4),
                    dbc.Col(dcc.RadioItems(id='forecast-metric-selector', options=[{'label': 'Sales Value ($)', 'value': 'sales'}, {'label': 'Sales Quantity (Units)', 'value': 'quantity'}], value='sales', inline=True), width="auto"),
                    dbc.Col([
                        html.P("Forecast Period (Days):", className="mb-0 small"),
                        dcc.Slider(id='forecast-period-slider', min=30, max=180, step=30, value=90, marks={i: str(i) for i in range(30, 181, 30)})
                    ], width=12, lg=4)
                ], className="mb-4 align-items-center"),
                dbc.Row(id='forecast-kpi-row', className="my-4 g-3"),
                dbc.Spinner(dcc.Graph(id='forecast-chart'), color="primary"),
                dbc.Spinner(dcc.Graph(id='forecast-decomposition-chart'), color="secondary")
            ]))
        ]),
    ])
], fluid=True, className="dbc")


# --- Helper Function & Data Callbacks ---
def parse_contents(contents, filename):
    if contents is None: return None
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename: df = pd.read_csv(io.StringIO(decoded.decode('utf-8', errors='ignore')))
        elif 'xls' in filename: df = pd.read_excel(io.BytesIO(decoded), sheet_name=None)
        else: return None
        if isinstance(df, dict): df = pd.concat(df.values(), ignore_index=True)
        df.columns = [str(col).strip().lower() for col in df.columns]
        return df
    except Exception as e: print(f"Error parsing {filename}: {e}"); return None

@app.callback(
    [Output('sales-data-store', 'data'), Output({'type': 'upload', 'index': 'sales'}, 'children')],
    Input({'type': 'upload', 'index': 'sales'}, 'contents'), State({'type': 'upload', 'index': 'sales'}, 'filename'))
def store_sales_data(c, f):
    if not c: raise PreventUpdate
    df = parse_contents(c, f)
    if df is None or 'itemname' not in df.columns: return None, html.Div([html.I(className="fas fa-times-circle text-danger me-2"), "Invalid Sales File"])
    return df.to_json(orient='split'), html.Div([html.I(className="fas fa-check-circle text-success me-2"), f])

@app.callback(
    [Output('reorder-data-store', 'data'), Output({'type': 'upload', 'index': 'reorder'}, 'children')],
    Input({'type': 'upload', 'index': 'reorder'}, 'contents'), State({'type': 'upload', 'index': 'reorder'}, 'filename'))
def store_reorder_data(c, f):
    if not c: raise PreventUpdate
    df = parse_contents(c, f)
    if df is None or 'sku' not in df.columns: return None, html.Div([html.I(className="fas fa-times-circle text-danger me-2"), "Invalid Reorder File"])
    return df.to_json(orient='split'), html.Div([html.I(className="fas fa-check-circle text-success me-2"), f])

@app.callback(
    [Output('margins-data-store', 'data'), Output({'type': 'upload', 'index': 'margins'}, 'children')],
    Input({'type': 'upload', 'index': 'margins'}, 'contents'), State({'type': 'upload', 'index': 'margins'}, 'filename'))
def store_margins_data(c, f):
    if not c: raise PreventUpdate
    df = parse_contents(c, f)
    if df is None or ('sku' not in df.columns and 'normal sku' not in df.columns): return None, html.Div([html.I(className="fas fa-times-circle text-danger me-2"), "Invalid Margins File"])
    return df.to_json(orient='split'), html.Div([html.I(className="fas fa-check-circle text-success me-2"), f])

@app.callback(
    [Output('import-list-store', 'data'), Output({'type': 'upload', 'index': 'import'}, 'children')],
    Input({'type': 'upload', 'index': 'import'}, 'contents'), State({'type': 'upload', 'index': 'import'}, 'filename'))
def store_import_list(c, f):
    if not c: raise PreventUpdate
    df = parse_contents(c, f)
    if df is None: return None, html.Div([html.I(className="fas fa-times-circle text-danger me-2"), "Invalid Import File"])
    skus = []
    if 'retail channel sku' in df.columns: skus.extend(df['retail channel sku'].dropna().astype(str).str.strip().str.lower())
    if 'vendor/ mask sku' in df.columns: skus.extend(df['vendor/ mask sku'].dropna().astype(str).str.strip().str.lower())
    if not skus: return None, html.Div([html.I(className="fas fa-times-circle text-danger me-2"), "No SKU columns found"])
    return list(set(skus)), html.Div([html.I(className="fas fa-check-circle text-success me-2"), f])

@app.callback(
    Output('merged-data-store', 'data'),
    [Input('sales-data-store', 'data'), Input('reorder-data-store', 'data'), Input('margins-data-store', 'data')])
def merge_data(sales_json, reorder_json, margins_json):
    if not sales_json: raise PreventUpdate
    sales_df = pd.read_json(sales_json, orient='split')
    reorder_df = pd.read_json(reorder_json, orient='split') if reorder_json else pd.DataFrame()
    margins_df = pd.read_json(margins_json, orient='split') if margins_json else pd.DataFrame()

    mpf_data = {
        'wayfair': 0.25, 'home depot': 0.30, 'bed bath & beyond': 0.32,
        'oj commerce': 0.28, 'vir venture': 0.28, 'uber bazaar': 0.28,
        'unbeatable': 0.28, 'amazon ca warehouse': 0.56, 'amazon vc dsv': 0.25,
        'amazon warehouse': 0.35, 'target': 0.30, 'lowes': 0.30, 'ashley home': 0.25
    }
    
    master_key, secondary_key = 'sku', 'normal sku'
    sales_df['original_itemname'] = sales_df['itemname']

    for df, cols in [(sales_df, ['itemname', 'customer name']), (reorder_df, [master_key, secondary_key, 'itemname']), (margins_df, [master_key, secondary_key])]:
        for col in cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
    
    sku_map = {}
    if not reorder_df.empty and secondary_key in reorder_df.columns:
        map_df = reorder_df.dropna(subset=[master_key, secondary_key])
        sku_map = pd.Series(map_df[master_key].values, index=map_df[secondary_key]).to_dict()

    sales_df[master_key] = sales_df['itemname'].map(sku_map).fillna(sales_df['itemname'])
    
    try: sales_df['orderdate'] = pd.to_datetime(sales_df['orderdate'], format='%d %B %Y', errors='coerce')
    except Exception: sales_df['orderdate'] = pd.to_datetime(sales_df['orderdate'], errors='coerce')
    sales_df = sales_df.dropna(subset=['orderdate'])
    sales_df = sales_df[sales_df['orderstatus'].isin(['Processed', 'Pending'])].copy()
    for col in ['quantity', 'price']: sales_df[col] = pd.to_numeric(sales_df[col], errors='coerce').fillna(0)
    sales_df['sales'] = sales_df['quantity'] * sales_df['price']
    
    merged_df = sales_df
    if not reorder_df.empty:
        agg_cols = {col: 'sum' for col in ['production', 'intransit'] if col in reorder_df.columns}
        first_cols = {col: 'first' for col in ['inv', 'image', 'category', 'product type', 'itemname'] if col in reorder_df.columns}
        reorder_agg = reorder_df.groupby(master_key).agg({**agg_cols, **first_cols}).reset_index()
        merged_df = pd.merge(merged_df, reorder_agg, on=master_key, how='left')

    if not margins_df.empty:
        if secondary_key in margins_df.columns:
            margins_df[master_key] = margins_df[secondary_key].map(sku_map).fillna(margins_df.get(master_key))
        cost_cols = [c for c in ['fob', 'landing cost'] if c in margins_df.columns]
        if cost_cols:
            for col in cost_cols: margins_df[col] = pd.to_numeric(margins_df[col], errors='coerce').fillna(0)
            margins_agg = margins_df.groupby(master_key)[cost_cols].first().reset_index()
            merged_df = pd.merge(merged_df, margins_agg, on=master_key, how='left', suffixes=('', '_margins'))
    
    if 'itemname' in merged_df.columns:
        merged_df['display_name'] = merged_df['itemname'].fillna(merged_df['original_itemname'])
    else:
        merged_df['display_name'] = merged_df['original_itemname']

    merged_df['mpf_percentage'] = merged_df['customer name'].map(mpf_data)
    merged_df['mpf_amount_per_unit'] = merged_df['price'] * merged_df['mpf_percentage']
    merged_df['profit_per_unit'] = merged_df['price'] - merged_df.get('landing cost', 0) - merged_df['mpf_amount_per_unit']
    merged_df['profit'] = merged_df['profit_per_unit'] * merged_df['quantity']
    merged_df['profit_margin'] = (merged_df['profit_per_unit'] / merged_df['price']).replace([np.inf, -np.inf], np.nan)

    return merged_df.to_json(date_format='iso', orient='split')

# --- Callbacks ---
@app.callback(
    [Output('main-dashboard-content', 'style'), Output('processing-status', 'children'),
     Output('customer-dropdown', 'options'), Output('date-range-picker', 'min_date_allowed'),
     Output('date-range-picker', 'max_date_allowed'), Output('date-range-picker', 'start_date'),
     Output('date-range-picker', 'end_date'), Output('forecast-item-dropdown', 'options'),
     Output('table-month-filter', 'options'), Output('profit-customer-dropdown', 'options'),
     Output('profit-category-dropdown', 'options')],
    Input('merged-data-store', 'data'))
def update_ui_elements(merged_json):
    if not merged_json: return {'display': 'none'}, "Waiting for Sales Data...", [], None, None, None, None, [], [], [], []
    df = pd.read_json(merged_json, orient='split')
    df['orderdate'] = pd.to_datetime(df['orderdate'])
    all_customers = sorted(df['customer name'].dropna().unique())
    all_categories = sorted(df['category'].dropna().unique()) if 'category' in df.columns else []
    min_date, max_date = df['orderdate'].min().date(), df['orderdate'].max().date()
    month_options = sorted(df['orderdate'].dt.to_period('M').unique().astype(str).tolist())
    item_options = [{'label': name, 'value': key} for name, key in df[['display_name', 'sku']].drop_duplicates().sort_values('display_name').values]
    return ({'display': 'block'}, "Data loaded successfully!", all_customers,
            min_date, max_date, min_date, max_date, item_options, month_options, all_customers, all_categories)

@app.callback(
    Output("upload-collapse", "is_open"),
    Input("collapse-upload-button", "n_clicks"), State("upload-collapse", "is_open"))
def toggle_upload_collapse(n, is_open):
    if n: return not is_open
    return is_open

@app.callback(
    [Output('item-dropdown', 'options'), Output('profit-item-dropdown', 'options')],
    [Input('merged-data-store', 'data'), Input('import-only-switch', 'value'), Input('import-list-store', 'data')])
def update_item_dropdown_options(merged_json, import_only, import_list):
    if not merged_json: return [], []
    df = pd.read_json(merged_json, orient='split')
    if import_only and import_list:
        df = df[df['sku'].isin(import_list)]
    options = [{'label': name, 'value': key} for name, key in df[['display_name', 'sku']].drop_duplicates().sort_values('display_name').values]
    return options, options

@app.callback(
    [Output('category-filter-col', 'style'), Output('category-dropdown', 'options'), Output('import-charts-section', 'style')],
    [Input('import-only-switch', 'value'), Input('item-dropdown', 'value'), Input('merged-data-store', 'data')])
def toggle_import_section(toggle_on, items_selected, merged_json):
    if not merged_json: raise PreventUpdate
    df = pd.read_json(merged_json, orient='split')
    categories = sorted(df['category'].dropna().unique()) if 'category' in df.columns else []
    if toggle_on and not items_selected:
        return {'display': 'block'}, [{'label': c, 'value': c} for c in categories], {'display': 'block'}
    return {'display': 'none'}, [], {'display': 'none'}

@app.callback(
    [Output('kpi-cards-row', 'children'), Output('monthly-quantity-chart', 'figure'),
     Output('customer-sales-chart', 'figure'), Output('main-data-table', 'columns'),
     Output('main-data-table', 'data'), Output('category-sales-pie-chart', 'figure'),
     Output('top-items-bar-chart', 'figure'), Output('price-trend-div', 'style'),
     Output('price-trend-chart', 'figure')],
    [Input('merged-data-store', 'data'), Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date'), Input('item-dropdown', 'value'),
     Input('customer-dropdown', 'value'), Input('import-only-switch', 'value'),
     Input('import-list-store', 'data'), Input('category-dropdown', 'value'),
     Input('table-month-filter', 'value')])
def update_main_visuals(merged_json, start_date, end_date, items, customers, import_only, import_list, categories, table_month):
    if not all([merged_json, start_date, end_date]): raise PreventUpdate
    df = pd.read_json(merged_json, orient='split')
    df['orderdate'] = pd.to_datetime(df['orderdate'])
    dff = df.copy()
    if import_only and import_list: dff = dff[dff['sku'].isin(import_list)]
    dff = dff[(dff['orderdate'] >= pd.to_datetime(start_date)) & (dff['orderdate'] <= pd.to_datetime(end_date))]
    if items: dff = dff[dff['sku'].isin(items)]
    if customers: dff = dff[dff['customer name'].isin(customers)]
    if categories: dff = dff[dff['category'].isin(categories)]
    
    total_sales, total_quantity, total_orders = dff['sales'].sum(), dff['quantity'].sum(), dff['orderid'].nunique()
    avg_order_value = total_sales / total_orders if total_orders > 0 else 0
    kpi_cards = [
        dbc.Col(dbc.Card(dbc.CardBody([html.H4("Total Sales"), html.H2(f"${total_sales:,.0f}")]))),
        dbc.Col(dbc.Card(dbc.CardBody([html.H4("Total Quantity"), html.H2(f"{total_quantity:,}")]))),
        dbc.Col(dbc.Card(dbc.CardBody([html.H4("Total Orders"), html.H2(f"{total_orders:,}")]))),
        dbc.Col(dbc.Card(dbc.CardBody([html.H4("Avg. Order Value"), html.H2(f"${avg_order_value:,.2f}")]))),
    ]

    fig_monthly = go.Figure(layout_title_text="Quantity Sold by Time")
    if not dff.empty:
        time_diff = pd.to_datetime(end_date) - pd.to_datetime(start_date)
        if time_diff.days < 14:
            rule, title = 'D', 'Daily Quantity Sold'
            grouped_data = dff.set_index('orderdate').resample(rule).agg({'quantity': 'sum', 'sales': 'sum'}).reset_index()
            grouped_data['period'] = grouped_data['orderdate'].dt.strftime('%d %b')
        elif time_diff.days <= 62:
            rule, title = 'W-Mon', 'Weekly Quantity Sold'
            grouped_data = dff.set_index('orderdate').resample(rule).agg({'quantity': 'sum', 'sales': 'sum'}).reset_index()
            grouped_data['period'] = grouped_data['orderdate'].dt.strftime('%d %b') + ' - ' + (grouped_data['orderdate'] + pd.to_timedelta(6, unit='d')).dt.strftime('%d %b, %Y')
        else:
            rule, title = 'M', 'Monthly Quantity Sold'
            grouped_data = dff.set_index('orderdate').resample(rule).agg({'quantity': 'sum', 'sales': 'sum'}).reset_index()
            grouped_data['period'] = grouped_data['orderdate'].dt.strftime('%B %Y')
        
        fig_monthly.add_trace(go.Bar(x=grouped_data['period'], y=grouped_data['quantity'], name='Quantity', text=grouped_data['quantity'], textposition='auto', customdata=grouped_data['sales']))
        fig_monthly.update_traces(hovertemplate='<b>Quantity</b>: %{y}<br><b>Sales</b>: $%{customdata[0]:,.2f}')
        fig_monthly.update_layout(title_text=title)
    
    fig_customer = go.Figure(layout_title_text='Top Customers by Quantity')
    if not dff.empty:
        customer_agg = dff.groupby('customer name').agg(quantity=('quantity', 'sum'), sales=('sales', 'sum')).nlargest(15, 'quantity').reset_index()
        fig_customer = px.bar(customer_agg, x='customer name', y='quantity', title='Top Customers by Quantity', text_auto=True, custom_data=['sales'])
        fig_customer.update_traces(hovertemplate='<b>Quantity</b>: %{y}<br><b>Sales</b>: $%{customdata[0]:,.2f}')
        fig_customer.update_xaxes(tickangle=-45)
    
    fig_pie, fig_top_items = go.Figure(layout_title_text="Sales by Category"), go.Figure(layout_title_text="Top 10 Items")
    if import_only and 'category' in dff.columns:
        dff_cat = dff[dff['category'].isin(categories)] if categories else dff
        if not dff_cat.empty:
            category_sales = dff_cat.groupby('category')['sales'].sum().reset_index()
            fig_pie = px.pie(category_sales, names='category', values='sales', title='Sales by Category')
            top_items_df = dff_cat.groupby('display_name').agg(sales=('sales', 'sum'), quantity=('quantity', 'sum')).nlargest(10, 'sales').reset_index()
            fig_top_items = px.bar(top_items_df, x='display_name', y='sales', title='Top 10 Items by Sales', custom_data=['quantity'])
            fig_top_items.update_traces(hovertemplate='<b>Sales</b>: $%{y:,.2f}<br><b>Quantity</b>: %{customdata[0]:,}<extra></extra>')
    
    dff_table = dff.copy()
    if table_month: dff_table = dff_table[dff_table['orderdate'].dt.to_period('M').astype(str) == table_month]
    display_cols = ['orderdate', 'display_name', 'customer name', 'quantity', 'price', 'sales', 'profit_margin']
    table_data = dff_table[display_cols].to_dict('records')
    for row in table_data: 
        row['orderdate'] = pd.to_datetime(row['orderdate']).strftime('%Y-%m-%d')
        row['profit_margin'] = f"{row.get('profit_margin', 0) * 100:.1f}%" if pd.notna(row.get('profit_margin')) else '-'
    table_cols = [{"name": "Date", "id": "orderdate"}, {"name": "Item Name", "id": "display_name"}, {"name": "Customer", "id": "customer name"}, {"name": "Quantity", "id": "quantity"}, {"name": "Price", "id": "price"}, {"name": "Sales", "id": "sales"}, {"name": "Profit Margin", "id": "profit_margin"}]
    
    price_trend_style, fig_price_trend = {'display': 'none'}, go.Figure()
    if items and customers:
        trend_df = dff[dff['sku'].isin(items) & dff['customer name'].isin(customers)]
        if not trend_df.empty:
            fig_price_trend = px.line(trend_df, x='orderdate', y='price', color='customer name', title='Price Trend Over Time', markers=True, custom_data=['profit_margin'])
            fig_price_trend.update_traces(hovertemplate='<b>Price</b>: $%{y:,.2f}<br><b>Profit Margin</b>: %{customdata[0]:.1%}<extra></extra>')
            price_trend_style = {'display': 'block'}

    return kpi_cards, fig_monthly, fig_customer, table_cols, table_data, fig_pie, fig_top_items, price_trend_style, fig_price_trend

@app.callback(
    Output('deep-dive-section', 'children'),
    [Input('item-dropdown', 'value'), Input('merged-data-store', 'data')])
def update_deep_dive(selected_items, merged_json):
    if not selected_items or not merged_json or len(selected_items) != 1: return None
    df = pd.read_json(merged_json, orient='split')
    item_data = df[df['sku'] == selected_items[0]].iloc[0]
    card_content = [
        dbc.Row([
            dbc.Col(html.Img(src=item_data.get('image', ''), style={'height':'150px', 'width': 'auto'}), width=3),
            dbc.Col([
                html.H4(item_data.get('display_name')),
                html.P(f"Category: {item_data.get('category', 'N/A')}"),
                html.P(f"Product Type: {item_data.get('product type', 'N/A')}")
            ], width=9)
        ]), html.Hr(),
        dbc.Row([
            dbc.Col(html.Div([html.P("Inventory"), html.H5(f"{item_data.get('inv', 0):,.0f}")])),
            dbc.Col(html.Div([html.P("In Production"), html.H5(f"{item_data.get('production', 0):,.0f}")])),
            dbc.Col(html.Div([html.P("In Transit"), html.H5(f"{item_data.get('intransit', 0):,.0f}")])),
            dbc.Col(html.Div([html.P("FOB Cost"), html.H5(f"${item_data.get('fob', 0):,.2f}")])),
            dbc.Col(html.Div([html.P("Landing Cost"), html.H5(f"${item_data.get('landing cost', 0):,.2f}")]))
        ], className="text-center")]
    return dbc.Card(dbc.CardBody(card_content), className="mb-3")

@app.callback(
    [Output('inventory-kpi-row', 'children'), Output('in-stock-percentage-chart', 'figure'),
     Output('abc-summary-table', 'children'), Output('at-risk-table', 'data'),
     Output('at-risk-table', 'columns'), Output('abc-analysis-data-store', 'data')],
    [Input('merged-data-store', 'data'), Input('inventory-import-only-toggle', 'value'),
     Input('import-list-store', 'data')])
def update_inventory_tab(merged_json, import_only, import_list):
    if not merged_json: raise PreventUpdate
    df = pd.read_json(merged_json, orient='split')
    df['orderdate'] = pd.to_datetime(df['orderdate'])
    if import_only and import_list: df = df[df['sku'].isin(import_list)]

    fig_in_stock = go.Figure(layout_title_text="No 'Category' or 'Inventory' data found.")
    abc_summary_table = dbc.Table()
    inventory_kpis, at_risk_data, at_risk_cols = [], [], []
    abc_data_json = None
    
    if 'inv' in df.columns and 'landing cost' in df.columns:
        inv_df = df.drop_duplicates(subset=['sku'])
        total_inv_value = (inv_df['inv'] * inv_df['landing cost']).sum()
        inventory_kpis.append(dbc.Col(dbc.Card(dbc.CardBody([html.H5("Total Inv. Value"), html.H4(f"${total_inv_value:,.0f}")])), width=6, lg=2, className="mb-2"))
        
        turnover_kpis_children = []
        if 'category' in inv_df.columns:
            cats = ['Hot', 'Top', 'Normal', 'New', 'Slow']
            for cat in cats:
                cat_df = inv_df[inv_df['category'] == cat]
                cat_sales_df = df[df['category'] == cat]
                cat_cogs = (cat_sales_df['quantity'] * cat_sales_df.get('landing cost', 0)).sum()
                cat_inv_value = (cat_df['inv'] * cat_df['landing cost']).sum()
                cat_turnover = cat_cogs / cat_inv_value if cat_inv_value > 0 else 0
                turnover_kpis_children.append(html.Tr([html.Td(cat), html.Td(f"{cat_turnover:.2f}")]))
                inv_value = (cat_df['inv'] * cat_df['landing cost']).sum()
                inventory_kpis.append(dbc.Col(dbc.Card(dbc.CardBody([html.H5(f"Inv. Value ({cat})"), html.H4(f"${inv_value:,.0f}")])), width=6, lg=2, className="mb-2"))
        inventory_kpis.append(dbc.Col(dbc.Card(dbc.CardBody([html.H5("Inv. Turnover"), dbc.Table(html.Tbody(turnover_kpis_children), bordered=False, striped=True, hover=True, size='sm')])), width=12, lg=2, className="mb-2"))

        if 'category' in df.columns:
            unique_items = df.drop_duplicates(subset=['sku'])
            category_stock = unique_items.groupby('category').agg(total_skus=('sku', 'nunique'), in_stock_skus=('inv', lambda x: (x > 0).sum())).reset_index()
            category_stock['in_stock_percent'] = (category_stock['in_stock_skus'] / category_stock['total_skus'] * 100).round(1)
            display_categories = ['Hot', 'Top', 'Normal', 'New', 'Slow']
            category_stock = category_stock[category_stock['category'].isin(display_categories)]
            if not category_stock.empty:
                fig_in_stock = px.bar(category_stock, x='category', y='in_stock_percent', title='In-Stock SKU % by Category', text='in_stock_percent', custom_data=['total_skus'], labels={'in_stock_percent': 'In-Stock %'})
                fig_in_stock.update_traces(texttemplate='%{text}%', textposition='outside', hovertemplate='<b>%{x}</b><br>In-Stock: %{y:.1f}%<br>Total SKUs: %{customdata[0]:,}<extra></extra>'); fig_in_stock.update_yaxes(range=[0, 110])

        abc_df = df.copy()
        if 'category' in abc_df.columns:
            abc_df = abc_df[abc_df['category'].isin(['Hot', 'Top', 'Normal', 'New', 'Slow'])]
        
        abc_df = abc_df.dropna(subset=['landing cost', 'quantity'])
        item_consumption = abc_df.groupby(['sku', 'display_name']).apply(lambda x: (x['quantity'] * x['landing cost']).sum()).sort_values(ascending=False).reset_index(name='consumption_value')
        if not item_consumption.empty and item_consumption['consumption_value'].sum() > 0:
            total_value = item_consumption['consumption_value'].sum()
            item_consumption['percent_of_total'] = item_consumption['consumption_value'] / total_value
            item_consumption['cumulative_percent'] = (item_consumption['consumption_value'].cumsum() / total_value) * 100
            def assign_abc_class(p): return 'A' if p <= 80 else 'B' if p <= 95 else 'C'
            item_consumption['class'] = item_consumption['cumulative_percent'].apply(assign_abc_class)
            abc_data_json = item_consumption.to_json(orient='split')
            class_counts = item_consumption['class'].value_counts().reset_index()
            
            table_header = [html.Thead(html.Tr([html.Th("Class"), html.Th("No. of SKUs")]))]
            table_body = [html.Tbody([html.Tr([html.Td(row['class']), html.Td(row['count'])]) for index, row in class_counts.iterrows()])]
            abc_summary_table = dbc.Table(table_header + table_body, bordered=True, striped=True, hover=True)

        risk_df = df.drop_duplicates(subset=['sku']).copy()
        for col in ['inv', 'intransit', 'production']: risk_df[col] = pd.to_numeric(risk_df[col], errors='coerce').fillna(0)
        risk_df = risk_df[risk_df['category'].isin(['Hot', 'Top', 'Normal', 'New', 'Slow'])]
        risk_df['total_supply'] = risk_df['inv'] + risk_df['intransit'] + risk_df['production']
        avg_daily_sales = df[df['orderdate'] > df['orderdate'].max() - pd.Timedelta(days=30)].groupby('sku')['quantity'].sum() / 30
        risk_df = pd.merge(risk_df, avg_daily_sales.rename('avg_daily_qty'), on='sku', how='left').fillna(0)
        risk_df['days_of_supply_left'] = risk_df.apply(lambda r: r['total_supply'] / r['avg_daily_qty'] if r['avg_daily_qty'] > 0 else np.inf, axis=1)
        at_risk_items = risk_df[risk_df['days_of_supply_left'] < 60].sort_values('days_of_supply_left')
        display_cols = {'display_name': 'Item Name', 'inv': 'In Stock', 'intransit': 'In Transit', 'production': 'In Prod.', 'days_of_supply_left': 'Days Left'}
        at_risk_data = at_risk_items.rename(columns=display_cols)[list(display_cols.values())].to_dict('records')
        for row in at_risk_data: row['Days Left'] = int(row['Days Left']) if np.isfinite(row['Days Left']) else 'inf'
        at_risk_cols = [{"name": i, "id": i} for i in display_cols.values()]
        
    return inventory_kpis, fig_in_stock, abc_summary_table, at_risk_data, at_risk_cols, abc_data_json

@app.callback(
    Output("download-abc-component", "data"),
    Input("download-abc-csv", "n_clicks"),
    [State("abc-analysis-data-store", "data"), State("merged-data-store", "data")],
    prevent_initial_call=True)
def download_abc_analysis(n_clicks, abc_json, merged_json):
    if not abc_json or not merged_json: raise PreventUpdate
    abc_df = pd.read_json(abc_json, orient='split')
    df = pd.read_json(merged_json, orient='split')
    
    df['orderdate'] = pd.to_datetime(df['orderdate'])
    
    risk_df = df.drop_duplicates(subset=['sku']).copy()
    for col in ['inv', 'intransit', 'production']: risk_df[col] = pd.to_numeric(risk_df[col], errors='coerce').fillna(0)
    risk_df['total_supply'] = risk_df['inv'] + risk_df['intransit'] + risk_df['production']
    avg_daily_sales = df[df['orderdate'] > df['orderdate'].max() - pd.Timedelta(days=30)].groupby('sku')['quantity'].sum() / 30
    risk_df = pd.merge(risk_df, avg_daily_sales.rename('avg_daily_qty'), on='sku', how='left').fillna(0)
    risk_df['days_of_supply_left'] = risk_df.apply(lambda r: r['total_supply'] / r['avg_daily_qty'] if r['avg_daily_qty'] > 0 else np.inf, axis=1)
    
    abc_df = pd.merge(abc_df, risk_df[['sku', 'inv', 'intransit', 'production', 'days_of_supply_left']], on='sku', how='left')

    output_cols = ['display_name', 'sku', 'class', 'consumption_value', 'percent_of_total', 'cumulative_percent', 'inv', 'intransit', 'production', 'days_of_supply_left']
    abc_df_to_export = abc_df[[col for col in output_cols if col in abc_df.columns]]
    
    header = """# ABC Analysis Data Explanation
# display_name: The name of the item for display.
# sku: The master unique identifier for the item.
# class: The ABC class (A, B, or C) assigned to the item based on consumption value (A=Top 80%, B=Next 15%, C=Bottom 5%).
# consumption_value: Total value consumed over the period (Quantity Sold * Landing Cost).
# percent_of_total: The item's percentage contribution to the total consumption value.
# cumulative_percent: The cumulative percentage of consumption value used for ranking.
# inv: Current inventory on hand.
# intransit: Inventory currently in transit.
# production: Inventory currently in production.
# days_of_supply_left: Estimated days of supply remaining based on the last 30 days of sales.
# --- Data Starts Below ---
"""
    
    csv_string = abc_df_to_export.to_csv(index=False)
    full_content = header + csv_string

    return dict(content=full_content, filename="abc_analysis_details.csv")

@app.callback(
    [Output('forecast-chart', 'figure'), Output('forecast-kpi-row', 'children'),
     Output('forecast-decomposition-chart', 'figure')],
    [Input('forecast-item-dropdown', 'value'), Input('merged-data-store', 'data'),
     Input('inventory-import-only-toggle', 'value'), Input('import-list-store', 'data'),
     Input('forecast-metric-selector', 'value'), Input('forecast-period-slider', 'value')])
def update_forecast_chart(selected_item_key, merged_json, import_only, import_list, metric, period):
    if not selected_item_key or not merged_json: 
        return go.Figure(layout_title_text="Select an item to generate its forecast"), [], go.Figure()
    if import_only and import_list and (selected_item_key not in import_list): 
        return go.Figure(layout_title_text="This is not an Import item."), [], go.Figure()
    
    df = pd.read_json(merged_json, orient='split')
    df['orderdate'] = pd.to_datetime(df['orderdate'])
    item_df = df[df['sku'] == selected_item_key].copy()
    item_name = item_df['display_name'].iloc[0] if not item_df.empty else selected_item_key
    
    df_prophet = item_df.groupby(item_df['orderdate'].dt.date)[metric].sum().reset_index()
    df_prophet.rename(columns={'orderdate': 'ds', metric: 'y'}, inplace=True)
    
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    if len(df_prophet) < 2: 
        return go.Figure(layout_title_text=f"Not enough data to forecast for {item_name}"), [], go.Figure()

    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True).fit(df_prophet)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Historical'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecasted', line={'dash': 'dash'}))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', name='Confidence Interval'))
    fig.update_layout(title=f'{period}-Day {metric.title()} Forecast for {item_name}', xaxis_title='Date', yaxis_title=metric.title())

    future_forecast = forecast[forecast['ds'] > df_prophet['ds'].max()]
    predicted_total = future_forecast['yhat'].sum()
    lower_bound = future_forecast['yhat_lower'].sum()
    upper_bound = future_forecast['yhat_upper'].sum()

    historical_rate = df_prophet['y'][-30:].mean()
    forecast_rate = future_forecast['yhat'].mean()
    growth_pct = ((forecast_rate / historical_rate) - 1) * 100 if historical_rate > 0 else 0

    kpi_cards = [
        dbc.Col(dbc.Card(dbc.CardBody([html.H5(f"Predicted Total {metric.title()}"), html.H4(f"{predicted_total:,.0f}")]))),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Expected Growth"), html.H4(f"{growth_pct:+.1f}%")]))),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Confidence Range"), html.H4(f"{lower_bound:,.0f} - {upper_bound:,.0f}")]))),
    ]

    fig_decomp = make_subplots(rows=2, cols=1, subplot_titles=('Trend', 'Weekly Seasonality'))
    fig_decomp.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', name='Trend'), row=1, col=1)
    weekly_data = forecast[['ds', 'weekly']].set_index('ds').resample('D').mean().ffill().reset_index()
    weekly_data['day_of_week'] = weekly_data['ds'].dt.day_name()
    weekly_seasonality = weekly_data.groupby('day_of_week')['weekly'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    fig_decomp.add_trace(go.Bar(x=weekly_seasonality.index, y=weekly_seasonality.values, name='Weekly'), row=2, col=1)
    fig_decomp.update_layout(title_text="Forecast Components")

    return fig, kpi_cards, fig_decomp

@app.callback(
    [Output('profit-kpi-row', 'children'),
     Output('profit-by-customer-chart', 'figure'),
     Output('top-profitable-items-chart', 'figure'),
     Output('margin-by-category-chart', 'figure')],
    [Input('merged-data-store', 'data'), Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date'), Input('profit-item-dropdown', 'value'),
     Input('profit-customer-dropdown', 'value'), Input('import-only-switch', 'value'),
     Input('import-list-store', 'data'), Input('profit-category-dropdown', 'value')])
def update_profitability_tab(merged_json, start_date, end_date, items, customers, import_only, import_list, categories):
    if not all([merged_json, start_date, end_date]): raise PreventUpdate
    df = pd.read_json(merged_json, orient='split')
    df['orderdate'] = pd.to_datetime(df['orderdate'])
    
    dff = df.copy()
    if import_only and import_list: dff = dff[dff['sku'].isin(import_list)]
    dff = dff[(dff['orderdate'] >= pd.to_datetime(start_date)) & (dff['orderdate'] <= pd.to_datetime(end_date))]
    if items: dff = dff[dff['sku'].isin(items)]
    if customers: dff = dff[dff['customer name'].isin(customers)]
    if categories: dff = dff[dff['category'].isin(categories)]

    total_profit = dff['profit'].sum()
    avg_margin = (dff['profit'].sum() / dff['sales'].sum()) * 100 if dff['sales'].sum() > 0 else 0
    profit_kpis = [
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Total Profit"), html.H4(f"${total_profit:,.0f}")]))),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Average Profit Margin"), html.H4(f"{avg_margin:.1f}%")]))),
    ]

    fig_profit_customer = go.Figure(layout_title_text="Profit by Customer")
    if not dff.empty:
        customer_profit = dff.groupby('customer name')['profit'].sum().nlargest(15).reset_index()
        fig_profit_customer.add_trace(go.Bar(x=customer_profit['customer name'], y=customer_profit['profit'], text=customer_profit['profit'], texttemplate='$%{y:,.0f}', textposition='auto'))
        fig_profit_customer.update_layout(title_text="Profit by Customer", xaxis_tickangle=-45)

    fig_top_profit_items = go.Figure(layout_title_text="Top 10 Most Profitable Items")
    if not dff.empty:
        item_profit = dff.groupby('display_name')['profit'].sum().nlargest(10).reset_index()
        fig_top_profit_items.add_trace(go.Bar(x=item_profit['display_name'], y=item_profit['profit'], text=item_profit['profit'], texttemplate='$%{y:,.0f}', textposition='auto'))
        fig_top_profit_items.update_layout(title_text="Top 10 Most Profitable Items")

    fig_margin_category = go.Figure(layout_title_text="Average Margin by Category")
    if not dff.empty and 'category' in dff.columns:
        margin_df = dff[~dff['category'].isin(['closeout', 'discontinued'])]
        category_profit = margin_df.groupby('category').agg(total_profit=('profit', 'sum'), total_sales=('sales', 'sum')).reset_index()
        category_profit['avg_margin'] = (category_profit['total_profit'] / category_profit['total_sales'] * 100)
        fig_margin_category.add_trace(go.Bar(x=category_profit['category'], y=category_profit['avg_margin'], text=category_profit['avg_margin'], texttemplate='%{y:.1f}%', textposition='auto'))
        fig_margin_category.update_layout(title_text="Average Margin % by Category")

    return profit_kpis, fig_profit_customer, fig_top_profit_items, fig_margin_category

@app.callback(
    Output("download-csv", "data"),
    Input("export-csv-button", "n_clicks"), State("main-data-table", "data"),
    prevent_initial_call=True)
def export_table_to_csv(n_clicks, table_data):
    if not table_data: raise PreventUpdate
    return dcc.send_data_frame(pd.DataFrame(table_data).to_csv, "filtered_transactions.csv", index=False)

@app.callback(
    [Output('item-dropdown', 'value', allow_duplicate=True), Output('customer-dropdown', 'value', allow_duplicate=True),
     Output('category-dropdown', 'value', allow_duplicate=True), Output('date-range-picker', 'start_date', allow_duplicate=True),
     Output('date-range-picker', 'end_date', allow_duplicate=True)],
    Input('clear-filters-button', 'n_clicks'), State('merged-data-store', 'data'),
    prevent_initial_call=True)
def clear_all_filters(n_clicks, merged_json):
    if not merged_json: raise PreventUpdate
    df = pd.read_json(merged_json, orient='split')
    df['orderdate'] = pd.to_datetime(df['orderdate'])
    min_date, max_date = df['orderdate'].min().date(), df['orderdate'].max().date()
    return [], [], [], min_date, max_date

if __name__ == '__main__':
    app.run(debug=False)
