#python3
import matplotlib.pyplot as plt
import networkx as nx
import chess.pgn as ch
import datetime
import pickle
import random

def read_data(file,limit):
    games = []
    pgn = open(file)
    # with open(file) as pgn:
    while True:
        game = ch.read_game(pgn)
        if not game or len(games)==limit:
            # this file is way too big hopefully this gives a meaningful enough result 
            break
        if game.headers['Event'] != 'FICS rated blitz game': continue
        if game.headers['TimeControl'] != '180+0': continue
        # for consistency, we only look at blitz games with time control value of 180+0
        games.append(game)
        print('read',len(games),'games so far...')
    pgn.close()
    return games

def create_timestamp(game):
    date = game.headers['Date']
    time = game.headers['Time']
    d = [int(da) for da in date.split('.')]
    t = [int(ti) for ti in time.split(':')]
    do = datetime.datetime(d[0],d[1],d[2],t[0],t[1],t[2])
    return do.timestamp()

def parse_header(game):
    # outputs a dictionary for the game header value
    data = {}
    for k,v in game.headers.items():
        data[k] = v
    return data

def sort_data(dataset):
    # for now let's have the first 60 percent as base data for prediction
    # and the rest as testing
    # to keep consistency, we also only look at 'FICS rated blitz game' game type
    temporal = []
    datas = {}
    for game in dataset:
        # d is a chess object. 
        # for now I will just deal with the data in headers,
        # which gives us the player names and game id and ratings
        time = create_timestamp(game)
        game_id = game.headers['FICSGamesDBGameNo']
        datas[game_id] = game
        temporal.append((time,game_id))
    # I have to use game id to sort because it doesn't like having game object in array while sorting
    temporal.sort()
    cutoff = int(len(temporal) * .6)
    base = temporal[:cutoff]    
    test = temporal[cutoff:]
    b = [parse_header(datas[ba[1]]) for ba in base]
    t = [parse_header(datas[te[1]]) for te in test]
    return b,t

def create_graph(dataset):
    # dataset is an array of data dictionary for each matchup
    G = nx.DiGraph()
    # instead of making a multigraph for every duplicate matchup, I update the weight (record) on the edge.
    # So if A loses to B twice and wins once, B -> A will have weight 2 and A -> B will have weight 1
    edgelist = {}
    for game in dataset:
        d = game
        # each player will be a node
        winner = d['Result']
        w_player = d['White']
        b_player = d['Black']
        # I'm just adding all the rest of the data in to the edge
        # the winner of the game is the source of the edge
        draw = False
        if winner == '1-0': players = (w_player,b_player)
        elif winner == '0-1': players = (b_player,w_player)
        else:
            draw = True
            # in case of a draw, we add edges for both direction
            # so that it basically counts as a win from both sides
            p1 = (w_player,b_player)
            p2 = (b_player,w_player)
        if not draw:
            if players not in edgelist.keys(): edgelist[players] = 1
            else: edgelist[players] += 1
        else:
            if p1 not in edgelist.keys(): edgelist[p1] = 1
            else: edgelist[p1] += 1
            if p2 not in edgelist.keys(): edgelist[p2] = 1
            else: edgelist[p2] += 1
    edges = [(players[0],players[1],{'weight':weight}) 
            for players,weight in edgelist.items()]
    G.add_edges_from(edges)
    print("I have",len(list(G.nodes())),'players in this graph')
    return G

def common_neighbors(white,black,g,threshold):
    beatenby_white = set(g.neighbors(white))
    beatenby_black = set(g.neighbors(black))
    # we do not consider players with record less than [threshold] games, since it is not enough data
    beat_white = set(g.predecessors(white))
    beat_black = set(g.predecessors(black))
    if len(beatenby_black) < threshold or len(beatenby_white) < threshold or len(beat_white) < threshold or len(beat_black) < threshold:
        # print('one of the players do not have enough records, skpping..')
        return 'SKIP'
    # A simple one
    whitewins = len(beatenby_white|beat_black)/len(beatenby_black|beat_white)
    predicted_winner = ''
    if whitewins > 1:
        # this player white has beaten more players who beat player black then the other way around
        predicted_winner = 'white'
    elif whitewins < 1:
        predicted_winner = 'black'
    else: return 'SKIP'
    # These guys also return the calculated values since I may want to use it later for combining outcomes
    # print('i am returning',(predicted_winner,whitewins))
    return predicted_winner

def mix_com_weight(white,black,g,threshold):
    comm_rate = common_neighbors(white,black,g,threshold)
    ew_rate = edge_weights(white,black,g,threshold)
    if comm_rate == 'SKIP' or ew_rate == 'SKIP': return 'SKIP'
    print('common_neighbor value is:',comm_rate[1],' edge_weight value is:',ew_rate[1])
    combined_rate = comm_rate[1] * ew_rate[1]
    if combined_rate > 1: return 'white'
    elif combined_rate < 1: return 'black'
    else: return 'SKIP'

def number_paths(white,black,g,threshold):
    # this guy is super slow right now!!
    w_to_b = len(list(nx.all_simple_paths(g,white,black,threshold)))
    b_to_w = len(list(nx.all_simple_paths(g,black,white,threshold)))
    # I should expand on this by weighing the edges differently for each path
    # such as if path has wins against higher ranks, give more weight
    # maybe I should have cutoff of something like 3 and calculate the 
    # average weight for the path to compare against 
    # if w_to_b < threshold or b_to_w < threshold: return 'SKIP'
    if w_to_b == 0 or b_to_w == 0: return 'SKIP'
    rate = w_to_b/b_to_w
    if rate > 1: return 'white'
    elif rate < 1: return 'black'
    else: return 'SKIP'

def edge_weights(white,black,g,threshold):
    w_victories = list(g.neighbors(white))
    w_losses = list(g.predecessors(white))
    if len(w_victories) < threshold or len(w_losses) < threshold: return 'SKIP'
    w_vic_score = sum([g.get_edge_data(white,wv)['weight'] for wv in w_victories])
    w_los_score = sum([g.get_edge_data(wl,white)['weight'] for wl in w_losses])
    w_rate = w_vic_score/w_los_score
    b_victories = list(g.neighbors(black))
    b_losses = list(g.predecessors(black))
    if len(b_victories) < threshold or len(b_losses) < threshold: return 'SKIP'
    b_vic_score = sum([g.get_edge_data(black,bv)['weight'] for bv in b_victories])
    b_los_score = sum([g.get_edge_data(bl,black)['weight'] for bl in b_losses])
    b_rate = b_vic_score/b_los_score
    if b_rate == 0 or w_rate == 0: return 'SKIP'
    rate = w_rate/b_rate
    if rate > 1: return 'white'
    elif rate < 1: return 'black'
    else: return 'SKIP'

def random_walk(source,g,num):
    visits = {}
    alpha = 0.1
    for v in g.nodes(): visits[v] = 0
    for i in range(num):
        # we want to use edges because calling successors 
        # returns the multi-edged ones only once,
        # but the edges show all the edges
        edges = list(g.edges(source))
        while random.random() > alpha and edges:
            d = random.choice(edges)[1]
            visits[d] += 1
            edges = list(g.edges(d))
    for v in g.nodes(): visits[v] /= num
    return visits

def coinflip(white,black,g,threshold):
    # I decided i want to compare this to a truly random guess
    if random.random() > .4: return 'white'
    else: return 'black'


def pagerank_easy(white,black,g,threshold):
    # do two walks for each white as source and black as source.
    v_white = random_walk(white,g,threshold)
    v_black = random_walk(black,g,threshold)
    if v_white[black] > v_black[white]: return 'white'
    elif v_white[black] < v_black[white]: return 'black'
    else: return 'SKIP'

def calculate_fairgoodness(g,threshold):
    # for this to work, I will first normalize all the weights
    # So i make a new graph with the edges and whatnot
    # This new graph has edges going both ways for every interaction
    # and the weight of the two edges between a pair of nodes will always sum to 1
    newG = nx.DiGraph()
    for u in g.nodes():
        for v in g.nodes():
            if u == v: continue
            if g.has_edge(u,v) and not newG.has_edge(u,v):
                w = g.get_edge_data(u,v)['weight']
                if g.has_edge(v,u):
                    w1 = g.get_edge_data(v,u)['weight']
                    newG.add_edge(u,v,weight=w/(w+w1))
                else:
                    newG.add_edge(u,v,weight=1)
                    newG.add_edge(v,u,weight=0)
    # initiate both fairness and goodness as 1 for all nodes
    starting = {'f': 1, 'g':1}
    values = {}
    # initiating
    for u in newG.nodes(): values[u] = starting
    def avg(lst): 
        if len(lst) > 0: return sum(lst)/len(lst) 
        else: return 0
    for i in range(threshold):
        for u in newG.nodes():
            values[u]['g'] = avg([values[v]['f'] * newG.get_edge_data(v,u)['weight'] for v in newG.predecessors(u)])
            values[u]['f'] = 1 - avg([newG.get_edge_data(u,v)['weight'] - values[v]['g'] for v in newG.neighbors(u)])
    return values

# def fairgoodness(white,black,)

def simulate(test,g,threshold, fn):
    correct_times = 0
    total_times = 0
    for data in test:
        matchup = data
        # matchup is the data of the matchup happening
        p_white = matchup['White']
        p_black = matchup['Black']
        white_rating = matchup['WhiteElo']
        black_rating = matchup['BlackElo']
        # make sure both players are recorded in the graph:
        if not (g.has_node(p_white) and g.has_node(p_black)):
            # print("one of the players do not have any record, skipping...")
            continue
        # here is when I start analyzing the players' past wins
        predicted_winner = fn(p_white,p_black,g,threshold)
        if predicted_winner == 'SKIP': continue
        predicted_outcome = ''
        if predicted_winner == 'white': predicted_outcome = '1-0'
        elif predicted_winner == 'black': predicted_outcome = '0-1'
        elif predicted_winner == 'draw': predicted_outcome = '1/2-1/2'
        # not enough data to make guess!
        # checking the actual winner here:
        if matchup['Result'] == predicted_outcome:
            print('guessed right!',predicted_winner,'wins!')
            correct_times += 1
        else:
            print('guess was wrong!',predicted_winner,'did not win!')
        total_times+=1
        print('rate so far:',round(correct_times/total_times * 100,2) ,'percent, at',total_times,'guesses')
    print('correct times:',correct_times)
    print('total guesses:', total_times)
    print('threshold was:',threshold)


#===================DRIVER CODE======================

# preparing data
'''This doesn't need to be done anymore, now i have 3 pickles to choose from'''
# datapaths = ['201911.pgn',
#             '201912.pgn',
#             '202001.pgn']
# # making pickels out of these graphs and test sets because its a pain to wait every time
# for path in datapaths:
#     print('working with dataset:',path)
#     prefix = path.split('.')[0]
#     dataset = read_data(path,100000)
#     base,test = sort_data(dataset)
#     g = create_graph(base)
#     nx.write_gpickle(g,prefix+'_graph')
#     outfile = open(prefix+'_test','ab')
#     pickle.dump(test,outfile)
#     outfile.close()

# function to change directed graph to directed multidigraph
def dgraph_to_mgraph(oldG):
    G = nx.MultiDiGraph()
    old_edges = list(oldG.edges)
    new_edges = []
    for ee in old_edges:
        w = oldG.get_edge_data(ee[0],ee[1])['weight']
        for i in range(w):
            new_edges.append((ee[0],ee[1]))
    G.add_edges_from(new_edges)
    return G


g1 = nx.read_gpickle('201911_graph')
mg1 = nx.read_gpickle('201911_mgraph')
d = open('201911_test','rb')
t1 = pickle.load(d)
d.close()

g2 = nx.read_gpickle('201912_graph')
mg2 = nx.read_gpickle('201912_mgraph')
d = open('201912_test','rb')
t2 = pickle.load(d)
d.close()

g3 = nx.read_gpickle('202001_graph')
mg3 = nx.read_gpickle('202001_mgraph')
d = open('202001_test','rb')
t3 = pickle.load(d)
d.close()

# simulate(t1,mg1,500,pagerank_easy)
simulate(t1,g1,1,coinflip)
# simulate(t1,g1,20,edge_weights)

# TODO: take in the opening moves as a factor in the prediction as well.
# I'm just gonna use the opening ECO number, 
# since the other stuff doesn't seem to add up very well and I also don't really know what it means

# TODO: So I've just now realized that this should be a multigraph as same players may play against each other again
# not sure how I'm going to deal with that.. maybe I should only look at paths that can be created in a temporal order?
# To address the above issue, I've decided to give the edges weights that are incremented for every duplicate victory.

# So far, from what I've seen with the results, seems that 50% was a bit too low of a threshold for a 'good' prediction.
# Even the common neighbors method yields about 56 percent accuracy, and so does the number paths method.

# And the number paths seems a bit inefficient, since it takes too long and the process killed itself when running on
# a dataset of 100,000 games, so I need to think about something that does not involve as much brute force and more metrics

# Observations so far:
# Obviously, for common neighbors method, the accuracy goes up as I increase the threshold.
# Although at some point, if the threshold is too high, it actually dropped as the dataset did not have enough
# entries satisfying the threshold.

# For a threshold of a 'good' prediction, I am shooting for higher than 70%, since it's pretty easy to guess over 50%

# Algorithm to try: Fairness/Goodness
