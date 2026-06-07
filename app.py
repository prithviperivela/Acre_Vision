import os
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# ── Load data ────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "6slider_parcel_scores.csv")
dashboard_df = pd.read_csv(DATA_PATH)
print(f"📊 Loaded {len(dashboard_df)} parcels for dashboard")

# ── Create the Dash app ─────────────────────────────────────
app = dash.Dash(__name__)
server = app.server  # Required for gunicorn

# ── Modern CSS styling ───────────────────────────────────────
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Hyderabad Land Suitability Analysis</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }

            .main-container {
                max-width: 1600px;
                margin: 0 auto;
                background: #ffffff;
                border-radius: 24px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                overflow: hidden;
            }

            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 40px 50px;
                color: white;
            }

            .header h1 {
                font-size: 2.5rem;
                font-weight: 700;
                margin: 0;
                letter-spacing: -0.5px;
            }

            .header p {
                font-size: 1.1rem;
                margin-top: 10px;
                opacity: 0.95;
                font-weight: 300;
            }

            .content-wrapper {
                display: flex;
                min-height: 800px;
            }

            .sidebar {
                width: 420px;
                background: #f8f9fa;
                padding: 40px 30px;
                border-right: 1px solid #e9ecef;
                overflow-y: auto;
            }

            .sidebar h3 {
                color: #1a1a1a;
                font-size: 1.4rem;
                font-weight: 600;
                margin-bottom: 30px;
                display: flex;
                align-items: center;
            }

            .slider-container {
                margin-bottom: 35px;
            }

            .slider-label {
                font-weight: 600;
                font-size: 1rem;
                margin-bottom: 12px;
                display: flex;
                align-items: center;
                color: #2d3748;
            }

            .slider-label span {
                margin-left: 8px;
            }

            .rc-slider {
                margin-bottom: 10px;
            }

            .rc-slider-track {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
                height: 6px !important;
            }

            .rc-slider-handle {
                border: 3px solid #667eea !important;
                background: white !important;
                width: 20px !important;
                height: 20px !important;
                margin-top: -7px !important;
                box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4) !important;
            }

            .rc-slider-handle:hover, .rc-slider-handle:active {
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.6) !important;
            }

            .rc-slider-rail {
                height: 6px !important;
                background: #e2e8f0 !important;
            }

            .weight-summary {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 16px;
                margin-top: 30px;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            }

            .weight-summary h4 {
                font-size: 1.2rem;
                margin-bottom: 20px;
                font-weight: 600;
            }

            .weight-summary p {
                margin: 10px 0;
                font-size: 0.95rem;
                display: flex;
                justify-content: space-between;
                padding: 8px 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            }

            .weight-summary p:last-child {
                border-bottom: none;
                font-weight: 600;
                font-size: 1.1rem;
                margin-top: 10px;
                padding-top: 15px;
                border-top: 2px solid rgba(255, 255, 255, 0.3);
            }

            .main-content {
                flex: 1;
                padding: 40px;
                background: white;
                overflow-y: auto;
            }

            .section-title {
                color: #1a1a1a;
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
            }

            .map-container {
                background: white;
                border-radius: 16px;
                padding: 20px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
                margin-bottom: 40px;
            }

            .table-container {
                background: white;
                border-radius: 16px;
                padding: 30px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            }

            table {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                font-size: 0.95rem;
            }

            thead {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }

            thead th {
                padding: 18px 15px;
                text-align: left;
                font-weight: 600;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            thead th:first-child {
                border-top-left-radius: 12px;
            }

            thead th:last-child {
                border-top-right-radius: 12px;
            }

            tbody tr {
                border-bottom: 1px solid #e9ecef;
                transition: all 0.2s ease;
            }

            tbody tr:hover {
                background: #f8f9fa;
                transform: scale(1.01);
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            }

            tbody td {
                padding: 16px 15px;
                color: #2d3748;
            }

            tbody tr:last-child {
                border-bottom: none;
            }

            .map-link {
                color: #667eea;
                text-decoration: none;
                font-weight: 600;
                padding: 8px 16px;
                background: #f0f4ff;
                border-radius: 8px;
                display: inline-block;
                transition: all 0.2s ease;
            }

            .map-link:hover {
                background: #667eea;
                color: white;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            }

            .score-badge {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 6px 12px;
                border-radius: 8px;
                font-weight: 600;
            }

            @media (max-width: 1200px) {
                .content-wrapper {
                    flex-direction: column;
                }

                .sidebar {
                    width: 100%;
                    border-right: none;
                    border-bottom: 1px solid #e9ecef;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ── Layout ───────────────────────────────────────────────────
app.layout = html.Div(className='main-container', children=[
    # Header
    html.Div(className='header', children=[
        html.H1('🏗️ Hyderabad Land Suitability Analysis'),
        html.P('Interactive 6-Slider Dashboard — Adjust weights to find optimal land parcels')
    ]),

    # Content wrapper
    html.Div(className='content-wrapper', children=[
        # Left Sidebar — Sliders
        html.Div(className='sidebar', children=[
            html.H3('🎚️ Smart Weight Adjustments'),

            # Vacancy Score Slider
            html.Div(className='slider-container', children=[
                html.Div(className='slider-label', children=[
                    html.Span('🏗️ Vacancy Score (Land Availability)')
                ]),
                dcc.Slider(
                    id='vacancy-weight',
                    min=0, max=40, step=5, value=20,
                    marks={i: {'label': f'{i}%', 'style': {'color': '#667eea', 'fontWeight': '600'}} for i in range(0, 41, 10)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ]),

            # Road Accessibility Slider
            html.Div(className='slider-container', children=[
                html.Div(className='slider-label', children=[
                    html.Span('🛣️ Road Accessibility')
                ]),
                dcc.Slider(
                    id='road-weight',
                    min=0, max=40, step=5, value=20,
                    marks={i: {'label': f'{i}%', 'style': {'color': '#764ba2', 'fontWeight': '600'}} for i in range(0, 41, 10)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ]),

            # Healthcare Access Slider
            html.Div(className='slider-container', children=[
                html.Div(className='slider-label', children=[
                    html.Span('🏥 Healthcare Access')
                ]),
                dcc.Slider(
                    id='healthcare-weight',
                    min=0, max=40, step=5, value=20,
                    marks={i: {'label': f'{i}%', 'style': {'color': '#667eea', 'fontWeight': '600'}} for i in range(0, 41, 10)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ]),

            # Education Access Slider
            html.Div(className='slider-container', children=[
                html.Div(className='slider-label', children=[
                    html.Span('🎓 Education Access')
                ]),
                dcc.Slider(
                    id='education-weight',
                    min=0, max=30, step=5, value=15,
                    marks={i: {'label': f'{i}%', 'style': {'color': '#764ba2', 'fontWeight': '600'}} for i in range(0, 31, 10)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ]),

            # Lifestyle Convenience Slider
            html.Div(className='slider-container', children=[
                html.Div(className='slider-label', children=[
                    html.Span('🍽️ Lifestyle Convenience')
                ]),
                dcc.Slider(
                    id='lifestyle-weight',
                    min=0, max=30, step=5, value=15,
                    marks={i: {'label': f'{i}%', 'style': {'color': '#667eea', 'fontWeight': '600'}} for i in range(0, 31, 10)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ]),

            # Essential Services Slider
            html.Div(className='slider-container', children=[
                html.Div(className='slider-label', children=[
                    html.Span('🛠️ Essential Services')
                ]),
                dcc.Slider(
                    id='essential-weight',
                    min=0, max=20, step=5, value=10,
                    marks={i: {'label': f'{i}%', 'style': {'color': '#764ba2', 'fontWeight': '600'}} for i in range(0, 21, 5)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ]),

            # Weight Summary
            html.Div(id='weight-summary', className='weight-summary')
        ]),

        # Right Main Content — Map and Table
        html.Div(className='main-content', children=[
            # Map Section
            html.H3('🗺️ Top Land Parcels — Hyderabad', className='section-title'),
            html.Div(className='map-container', children=[
                dcc.Graph(id='parcel-map', style={'height': '500px'})
            ]),

            # Table Section
            html.H3('🏆 Top 10 Recommended Parcels', className='section-title'),
            html.Div(id='top-parcels-table', className='table-container')
        ])
    ])
])


# ── Callback ─────────────────────────────────────────────────
@app.callback(
    [Output('weight-summary', 'children'),
     Output('parcel-map', 'figure'),
     Output('top-parcels-table', 'children')],
    [Input('vacancy-weight', 'value'),
     Input('road-weight', 'value'),
     Input('healthcare-weight', 'value'),
     Input('education-weight', 'value'),
     Input('lifestyle-weight', 'value'),
     Input('essential-weight', 'value')]
)
def update_dashboard(vacancy_w, road_w, healthcare_w, education_w, lifestyle_w, essential_w):
    # Normalize weights to sum to 100%
    total = vacancy_w + road_w + healthcare_w + education_w + lifestyle_w + essential_w
    if total == 0:
        vacancy_w, road_w, healthcare_w, education_w, lifestyle_w, essential_w = 20, 20, 20, 15, 15, 10
        total = 100

    weights = {
        'vacancy': vacancy_w / total,
        'road_access': road_w / total,
        'healthcare': healthcare_w / total,
        'education': education_w / total,
        'lifestyle': lifestyle_w / total,
        'essential_services': essential_w / total
    }

    # Recalculate composite score with new weights
    dashboard_df['composite_score_dynamic'] = (
        weights['vacancy'] * (dashboard_df['vacancy_subscore'] / 100) +
        weights['road_access'] * (dashboard_df['road_subscore'] / 100) +
        weights['healthcare'] * (dashboard_df['healthcare_subscore'] / 100) +
        weights['education'] * (dashboard_df['education_subscore'] / 100) +
        weights['lifestyle'] * (dashboard_df['lifestyle_subscore'] / 100) +
        weights['essential_services'] * (dashboard_df['essential_services_subscore'] / 100)
    ) * 100

    # Get top 10 parcels
    top_10 = dashboard_df.nlargest(10, 'composite_score_dynamic')

    # Weight summary
    weight_summary = html.Div([
        html.H4('🎯 Current Weight Distribution'),
        html.P([html.Span('🏗️ Vacancy'), html.Span(f'{weights["vacancy"]*100:.1f}%')]),
        html.P([html.Span('🛣️ Road Access'), html.Span(f'{weights["road_access"]*100:.1f}%')]),
        html.P([html.Span('🏥 Healthcare'), html.Span(f'{weights["healthcare"]*100:.1f}%')]),
        html.P([html.Span('🎓 Education'), html.Span(f'{weights["education"]*100:.1f}%')]),
        html.P([html.Span('🍽️ Lifestyle'), html.Span(f'{weights["lifestyle"]*100:.1f}%')]),
        html.P([html.Span('🛠️ Essential Services'), html.Span(f'{weights["essential_services"]*100:.1f}%')]),
        html.P([html.Span('📊 Total'), html.Span('100.0%')])
    ])

    # Map
    fig = px.scatter_mapbox(
        top_10,
        lat="latitude",
        lon="longitude",
        hover_name="gid",
        hover_data={
            "composite_score_dynamic": ":.2f",
            "vacancy_score": ":.2f",
            "healthcare_count_1km": True,
            "lifestyle_count_1km": True,
            "latitude": False,
            "longitude": False
        },
        color="composite_score_dynamic",
        color_continuous_scale=[[0, "#667eea"], [0.5, "#764ba2"], [1, "#f093fb"]],
        size_max=15,
        zoom=10,
        height=500
    )

    fig.update_layout(
        mapbox_style="carto-positron",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family="Inter, sans-serif", size=12, color="#2d3748"),
        coloraxis_colorbar=dict(
            title="Score",
            thicknessmode="pixels",
            thickness=15,
            lenmode="pixels",
            len=300,
            yanchor="middle",
            y=0.5
        )
    )

    # Table
    table_rows = []
    for idx, (_, row) in enumerate(top_10.iterrows(), 1):
        table_rows.append(html.Tr([
            html.Td(f'#{idx}', style={'fontWeight': '700', 'color': '#667eea'}),
            html.Td(f'Parcel {row["gid"]}'),
            html.Td(html.Span(f'{row["composite_score_dynamic"]:.2f}', className='score-badge')),
            html.Td(f'{row["vacancy_score"]:.2f}%'),
            html.Td(str(int(row["healthcare_count_1km"]))),
            html.Td(str(int(row["lifestyle_count_1km"]))),
            html.Td(html.A('📍 View on Map',
                           href=row['google_maps_link'],
                           target="_blank",
                           className='map-link'))
        ]))

    table = html.Table([
        html.Thead(html.Tr([
            html.Th('Rank'),
            html.Th('Parcel ID'),
            html.Th('Score'),
            html.Th('Vacancy'),
            html.Th('Healthcare'),
            html.Th('Lifestyle'),
            html.Th('Location')
        ])),
        html.Tbody(table_rows)
    ])

    return weight_summary, fig, table


# ── Run ──────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host='0.0.0.0', port=port)
