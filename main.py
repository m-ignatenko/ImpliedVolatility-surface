import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
from scipy.interpolate import griddata

st.title('Implied Volatility Surface Visualizer')
user_ticker = st.text_input('Stock Ticker', 'SPY').upper()
display_mode = st.radio("Display Mode", ['Strike', 'Moneyness'])
spot_price = yf.Ticker(user_ticker).history()['Close'].iloc[-1]
@st.cache_data(ttl=3600)  
def get_option_data(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        expirations = ticker_obj.options
        
        if not expirations:
            st.error(f"No option data available for ticker {ticker}")
            return None
            
        time_to_expirations = []
        for i in expirations:
            d = datetime.strptime(i, "%Y-%m-%d").date()
            today = datetime.today().date()
            time_to_expirations.append(round((d-today).days/365, 2))
        
        option_data = []
        for ed in range(len(expirations)):
            try:
                option_chain = ticker_obj.option_chain(expirations[ed])
                calls = option_chain.calls
                
                for i, r in calls.iterrows():
                    if pd.notna(r['impliedVolatility']):
                        option_data.append({
                            'time_to_expiration': time_to_expirations[ed],
                            'expirationDate': expirations[ed],
                            'strike': r['strike'],
                            'moneyness': r['strike'] / spot_price,
                            'mid': (r['bid'] + r['ask'])/2,
                            'iv': r['impliedVolatility']
                        })
            except Exception as e:
                st.warning(f"Could not retrieve data for expiration {expirations[ed]}: {str(e)}")
                continue
                
        if not option_data:
            st.error(f"No valid option data found for ticker {ticker}")
            return None
            
        return pd.DataFrame(option_data)
        
    except Exception as e:
        st.error(f"Error retrieving data for ticker {ticker}: {str(e)}")
        return None

option_data = get_option_data(user_ticker)

if option_data is not None:
    st.subheader(f'Implied Volatility Surface for {user_ticker}')
    st.write(f"Data as of {datetime.today().date()}")
    
    x = option_data['time_to_expiration'].values  # Years to expiration
    if display_mode == 'Strike':
        y = option_data['strike'].values             # Strike prices
    else: y = option_data['moneyness'].values  #Moneyness
    z = option_data['iv'].values                 # Implied volatility

    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    zi = np.maximum(zi, 0)  #

    fig = go.Figure(data=[
        go.Surface(
            x=xi,
            y=yi,
            z=zi,
            colorscale='Balance',
            opacity=0.9,
            contours={
                "x": {"show": True, "color": "grey"},
                "y": {"show": True, "color": "grey"},
                "z": {"show": True, "start": 0, "color": "grey"}  
            },
            connectgaps=True,
            showscale=True
        )
    ])

    fig.add_trace(
        go.Surface(
            x=xi,
            y=yi,
            z=np.zeros_like(zi), 
            colorscale=[[0, 'rgba(0,0,0,0.1)'], [1, 'rgba(0,0,0,0.1)']],
            showscale=False,
            opacity=0.3,
            hoverinfo='skip'
        )
    )
    fig.update_layout(
        title=f'Implied Volatility Surface for {user_ticker}',
        scene=dict(
            xaxis_title='Years to Expiration',
            yaxis_title=display_mode,
            zaxis_title='Implied Volatility',
            zaxis=dict(range=[0, max(zi.flatten())*1.1]),  
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=800
    )
    fig.update_traces(
        hovertemplate=(
            "Years: %{x:.2f}<br>"
            "Strike: %{y:.2f}<br>"
            "IV: %{z:.2f}<extra></extra>"
        )
    )
    st.plotly_chart(fig, use_container_width=True)
