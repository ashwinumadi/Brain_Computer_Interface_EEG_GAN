import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input,Output
import dash_bootstrap_components as dbc
from app import app
import functions
import base64
import os
import time
from apps import app1
from flask_caching import Cache
cache = Cache(app.server, config={'CACHE_TYPE': 'simple'})

card2 = [
    dbc.CardBody([html.H1('BCI for Image Retrival', style = {'textAlign':'center', 'color':'black',}),
        
    ])
]

card1 = [
    dbc.CardBody([
        dbc.CardLink("Reconstruct", id = 'plot_b', n_clicks = 0, href="/app/app3", style = {'color':'black', 'textAlign':'center'}),
    ])
]

'''def update_plotted():
    if(os.path.exists("./assets/blah.png")):
        plotted = base64.b64encode(open("./assets/blah.png", 'rb').read()).decode("ascii")
    elif(os.path.exists("./assets/blah1.png")):
        plotted = base64.b64encode(open("./assets/blah1.png", 'rb').read()).decode("ascii")
    return plotted'''

def get_plot():
    

    file_name = "./flask1/app/repo/storage.txt"
    f = open(file_name, 'r')
    shapes = f.readline()
    num = f.readline()
    num = int(num)
    num-=1
    num = str(num)
    f.close()
    print(num)
    string = "./assets/plot" + num + ".png"
    plotted = base64.b64encode(open(string, 'rb').read()).decode("ascii")
    return plotted
    
#new_plot = update_plotted()

layout1 = html.Div([
    html.Div([dbc.Card(card2, color = 'pink')]),
    html.H1(id = 'kushal'),
    html.Br(),
    html.Img(id = "image", src = "data:image/png;base64,{}".format(get_plot()), style = {'margin-left':'32%'}),
    html.Br(),
    dcc.Store(id='s', data = [{'shape':'Circle', 'color':'Green'}]),
    html.Div([dbc.Card(card1, color = 'pink', outline = True, style = {'textAlign':'center', 'fontSize':35})],style = {'width':300, 'margin-left':680, 'margin-top':40}),
    
],

'''style = {
    'backgroundColor': 'black',
    'backgroundSize' : '100% 100%',
    'position': 'fixed',
    'min-height': '100%',
    'min-width': '100%',
}    '''
)



'''@app.callback(Output('s', 'data'),
    Input('plot_b', 'n_clicks'),
    )
def update_reconstruct(dog):
    print("Innond agthide")
    if(dog != 0):
        print("Inside if")
        functions.reconstruct()'''
        

