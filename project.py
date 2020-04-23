#python3
import networkx as nx
import chess.pgn as ch
import datetime
import pickle
import random
from tqdm import tqdm

'''
How to use:

The code for creating the training set/running the test is at the bottom of this file.
The inputs for the simulation can be changed in lines down there.
Once those lines are ready, just simply run this file and it will show results from the simulation 
'''

############## Graph Generation! (Doesn't need to be called anymore) ##############

def read_data(file,limit):
    games = []
    pgn = open(file)
    # with open(file) as pgn:
    for i in tqdm(range(limit)):
        game = ch.read_game(pgn)
        if game.headers['Event'] != 'FICS rated blitz game': continue
        if game.headers['TimeControl'] != '180+0': continue
        # for consistency, we only look at blitz games with time control value of 180+0
        games.append(parse_header(game))
    pgn.close()
    return games

def create_timestamp(game):
    date = game['Date']
    time = game['Time']
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

def sort_data(dataset,threshold):
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
        game_id = game['FICSGamesDBGameNo']
        datas[game_id] = game
        temporal.append((time,game_id))
    # I have to use game id to sort because it doesn't like having game object in array while sorting
    temporal.sort()
    cutoff = int(len(temporal) * threshold)
    base = temporal[:cutoff]    
    test = temporal[cutoff:]
    b = [datas[ba[1]] for ba in base]
    t = [datas[te[1]] for te in test]
    return b,t

def create_graph(dataset):
    # dataset is an array of data dictionary for each matchup
    G = nx.DiGraph()
    # instead of making a multigraph for every duplicate matchup, I update the weight (record) on the edge.
    # So if A loses to B twice and wins once, B -> A will have weight 2 and A -> B will have weight 1
    edgelist = {}
    print('creating graph...')
    for i in range(len(dataset)):
        d = dataset[i]
        # each player will be a node
        winner = d['Result']
        w_player = d['White']
        b_player = d['Black']
        # I'm just adding all the rest of the data in to the edge
        # the winner of the game is the source of the edge
        draw = False
        whitewin = False
        if winner == '1-0': 
            players = (w_player,b_player)
            whitewin = True
        elif winner == '0-1': 
            players = (b_player,w_player)
        else:
            draw = True
            # in case of a draw, we add edges for both direction
            # so that it basically counts as a win from both sides
            p1 = (w_player,b_player)
            p2 = (b_player,w_player)
        if not draw:
            if players not in edgelist.keys(): 
                if whitewin: edgelist[players] = {'weight':1,'whitewin':1,'blackwin':0}
                else: edgelist[players] = {'weight':1,'whitewin':0,'blackwin':1}
            else:
                edgelist[players]['weight'] += 1
                if whitewin: edgelist[players]['whitewin'] += 1
                else: edgelist[players]['blackwin'] += 1
        else:
            if p1 not in edgelist.keys(): edgelist[p1] = {'weight':1,'whitewin':.5,'blackwin':.5}
            else: 
                edgelist[p1]['weight'] += 1
                edgelist[p1]['whitewin'] += .5
                edgelist[p1]['blackwin'] += .5
            if p2 not in edgelist.keys(): edgelist[p2] = {'weight':1,'whitewin':.5,'blackwin':.5}
            else: 
                edgelist[p2]['weight'] += 1
                edgelist[p2]['whitewin'] += .5
                edgelist[p2]['blackwin'] += .5
    edges = [(players[0],players[1],{'weight':values['weight'],'whitewin':values['whitewin'],'blackwin':values['blackwin']}) 
            for players,values in edgelist.items()]
    G.add_edges_from(edges)
    print("I have",len(list(G.nodes())),'players in this graph')
    return G

###################################################################################


##################### Different methods used for predictions! #####################
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
    if whitewins > 1:
        # this player white has beaten more players who beat player black then the other way around
        return 'white'
    elif whitewins < 1:
        return 'black'
    else: return 'draw'

def com_neigh_consider_side(white,black,g,threshold):
    # players the white player has won against more time as white
    white_beat_as_white = []
    black_beat_as_black = []
    for p in list(g.neighbors(white)):
        game = g.get_edge_data(white,p)
        if game['whitewin'] >= game['blackwin']: white_beat_as_white.append(p)
    for p in list(g.neighbors(black)):
        game = g.get_edge_data(black,p)
        if game['blackwin'] >= game['whitewin']: black_beat_as_black.append(p)
    
    white_lost_as_white = []
    black_lost_as_black = []

    for p in list(g.predecessors(white)):
        game = g.get_edge_data(p,white)
        # if the opponent wins as black, means that this player lost as white
        if game['blackwin'] >= game['whitewin']: white_lost_as_white.append(p)
    for p in list(g.predecessors(black)):
        game = g.get_edge_data(p,black)
        if game['whitewin'] >= game['blackwin']: black_lost_as_black.append(p)
    
    wbw = set(white_beat_as_white)
    bbb = set(black_beat_as_black)

    wlw = set(white_lost_as_white)
    blb = set(black_lost_as_black)

    # this means that white is more likely to win as white than black is as black
    if len(wbw|blb) > len(bbb|wlw): return 'white'
    elif len(wbw|blb) < len(bbb|wlw): return 'black'
    else: return 'draw'

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
    else: return 'draw'
# basically a win rate comparison from training data
def edge_weights(white,black,g,threshold):
    w_victories = list(g.neighbors(white))
    w_losses = list(g.predecessors(white))
    if len(w_victories) < threshold or len(w_losses) < threshold: return 'SKIP'
    w_vic_score = sum([g.get_edge_data(white,wv)['weight'] for wv in w_victories])
    w_los_score = sum([g.get_edge_data(wl,white)['weight'] for wl in w_losses])
    w_rate = w_vic_score/(w_vic_score+w_los_score)
    b_victories = list(g.neighbors(black))
    b_losses = list(g.predecessors(black))
    if len(b_victories) < threshold or len(b_losses) < threshold: return 'SKIP'
    b_vic_score = sum([g.get_edge_data(black,bv)['weight'] for bv in b_victories])
    b_los_score = sum([g.get_edge_data(bl,black)['weight'] for bl in b_losses])
    b_rate = b_vic_score/(b_vic_score+b_los_score)
    if b_rate == 0 or w_rate == 0: return 'SKIP'
    rate = w_rate/b_rate
    if rate > 1: return 'white'
    elif rate < 1: return 'black'
    else: return 'draw'

# essentially a win rate for the player as the color they have
def edge_weights_consider_side(white,black,g,threshold):
    w_vic_as_w = [g.get_edge_data(white,p)['whitewin'] for p in g.neighbors(white)]
    w_los_as_w = [g.get_edge_data(p,white)['blackwin'] for p in g.predecessors(white)]

    b_vic_as_b = [g.get_edge_data(black,p)['blackwin'] for p in g.neighbors(black)]
    b_los_as_b = [g.get_edge_data(p,black)['whitewin'] for p in g.predecessors(black)]

    w_rate = w_vic_as_w/(w_vic_as_w+w_los_as_w)
    b_rate = b_vic_as_b/(b_vic_as_b+b_los_as_b)

    if b_rate == 0 and not w_rate == 0: return 'white'
    elif w_rate==0: return 'black'

    if w_rate > b_rate: return 'white'
    elif b_rate > w_rate: return 'black'
    else: return 'draw'

def random_walk(source,g,num):
    visits = {}
    alpha = 0.1
    for v in g.nodes(): visits[v] = 0
    for i in range(num):
        # we want to use edges because calling successors 
        # returns the multi-edged ones only once,
        # but the edges show all the edges
        d = source
        edges = list(g.edges(source))
        while random.random() > alpha and edges:
            # stops either when the random value is lower than alpha
            # or when there are no more edges to take
            d = random.choice(edges)[1]
            edges = list(g.edges(d))
        visits[d] += 1
    for v in g.nodes(): visits[v] /= num
    return visits

def pagerank_easy(white,black,g,num_iter):
    global stored_pagerank
    # do two walks for each white as source and black as source.
    if not white in stored_pagerank: 
        v_white = random_walk(white,g,num_iter)
        stored_pagerank[white] = v_white
    else:
        v_white = stored_pagerank[white]

    if not black in stored_pagerank: 
        v_black = random_walk(black,g,num_iter)
        stored_pagerank[black] = v_black
    else:
        v_black = stored_pagerank[black]

    if v_white[black] > v_black[white]: return 'white'
    elif v_white[black] < v_black[white]: return 'black'
    else: return 'draw'

def calculate_fairgoodness(oldG,threshold):
    import copy
    g = copy.deepcopy(oldG)
    # for this method to work we need back edges for all edge
    # g is a multidigraph, just like for the pageranks
    data = {}
    for e in g.edges: data[e] = {'weight':1}
    nx.set_edge_attributes(g,data)
    toadd = [(e[1],e[0]) for e in g.edges]
    g.add_edges_from(toadd,weight=-1)
    # now all the original edges in g have weight 1 
    # and the back edges have weight 0
    vals = {}
    # initiating values 
    for v in g.nodes: 
        vals[v] = {}
        vals[v]['f'] = 1
        vals[v]['g'] = 1
    print('prepping data...')
    for i in tqdm(range(threshold)):
        for v in g.nodes:
            # turn it into set first to get rid of duplicates
            # because i handle that while looping later
            in_edges = list(set(g.in_edges(v)))
            out_edges = list(set(g.edges(v)))
            # (Weight of outgoing edge) - (goodness of that edge being judged)            
            fair = []
            good = []
            for e in out_edges:
                ed = g.get_edge_data(e[0],e[1])
                for k,w in ed.items():
                    fair.append(abs(w['weight']- vals[e[1]]['g']))
            
            # (fairness of judging node) * (weight of the edge from that node)
            for e in in_edges:
                ed = g.get_edge_data(e[0],e[1])
                for k,w in ed.items():
                    good.append(vals[e[0]]['f'] * w['weight'])

            vals[v]['f'] = 1 - sum(fair)/(len(fair) * 2)
            vals[v]['g'] = sum(good)/len(good)
    return vals

def coinflip(white,black,g,threshold):
    # I decided i want to compare this to a truly random guess
    if random.random() < .5: return 'white'
    else: return 'black'

def fairgoodness(white,black,g,threshold):
    global fg
    w_fair = fg[white]['f']
    w_good = fg[white]['g']
    b_fair = fg[black]['f']
    b_good = fg[black]['g']
    w_to_b = w_fair * b_good
    b_to_w = b_fair * w_good
    if w_to_b > b_to_w: return 'white'
    elif b_to_w > w_to_b: return 'black'
    else: return 'DRAW'

###################################################################################

################## Actual Simulation and some helper functions!####################

def simulate(test,g,threshold, fn):
    correct_times = 0
    predicted_whitewin = 0
    predicted_blackwin = 0
    predicted_draw = 0
    correct_whitewin = 0
    correct_blackwin = 0
    correct_draw = 0 
    total_times = 0
    for i in tqdm(range(len(test))):
        matchup = test[i]
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
        if predicted_winner == 'white': 
            predicted_whitewin += 1
            predicted_outcome = '1-0'
        elif predicted_winner == 'black': 
            predicted_blackwin += 1
            predicted_outcome = '0-1'
        elif predicted_winner == 'draw': 
            predicted_draw += 1
            predicted_outcome = '1/2-1/2'
        # not enough data to make guess!
        # checking the actual winner here:
        if matchup['Result'] == predicted_outcome:
            # print('guessed right!',predicted_winner,'wins!')
            if predicted_winner == 'white':
                correct_whitewin += 1
            elif predicted_winner == 'black':
                correct_blackwin += 1
            else:
                correct_draw += 1
            correct_times += 1
        # else:
            # print('guess was wrong!',predicted_winner,'did not win!')
        total_times+=1
        # print('rate so far:',round(correct_times/total_times * 100,2) ,'percent, at',total_times,'guesses')
    print('total accuracy rate:   ',round(correct_times/total_times * 100,2),'%')
    print('accuracy for white win:',round(correct_whitewin/predicted_whitewin * 100,2),'%')
    print('accuracy for black win:',round(correct_blackwin/predicted_blackwin * 100,2),'%')
    print('accuracy for draw:     ',round(correct_draw/predicted_draw * 100,2),'%')
    print('correct times:         ',correct_times)
    print('total guesses:         ', total_times)
    print('threshold was:         ',threshold)

def read_prepped_data(smaller):
    prefix = 'data_200000/'
    if smaller:
        prefix = 'data_100000/'
    g1 = nx.read_gpickle(prefix+'201911_graph')
    mg1 = nx.read_gpickle(prefix+'201911_mgraph')
    d = open(prefix+'201911_test','rb')
    t1 = pickle.load(d)
    d.close()

    g2 = nx.read_gpickle(prefix+'201912_graph')
    mg2 = nx.read_gpickle(prefix+'201912_mgraph')
    d = open(prefix+'201912_test','rb')
    t2 = pickle.load(d)
    d.close()

    g3 = nx.read_gpickle(prefix+'202001_graph')
    mg3 = nx.read_gpickle(prefix+'202001_mgraph')
    d = open(prefix+'202001_test','rb')
    t3 = pickle.load(d)
    d.close()

    return g1,mg1,t1,g2,mg2,t2,g3,mg3,t3

# function to change directed graph to directed multidigraph
def dgraph_to_mgraph(oldG):
    G = nx.MultiDiGraph()
    old_edges = list(oldG.edges)
    new_edges = []
    for ee in old_edges:
        whitewincount = oldG.get_edge_data(ee[0],ee[1])['whitewin']
        w = oldG.get_edge_data(ee[0],ee[1])['weight']
        for i in range(w):
            if i < whitewincount: new_edges.append((ee[0],ee[1],{'win_as':'white'}))
            else: new_edges.append((ee[0],ee[1],{'win_as':'black'}))
    G.add_edges_from(new_edges)
    return G
    

#================================== DRIVER CODE ==================================#

# preparing data
'''This doesn't need to be done anymore, now i have 3 pickles to choose from
    Uncomment the following block the first time if the pickle files are broken for whatever reason'''
'''
datapaths = ['201911.pgn',
            '201912.pgn',
            '202001.pgn']
# making pickles out of these graphs and test sets because its a pain to wait every time
for path in datapaths:
    print('working with dataset:',path)
    prefix = path.split('.')[0]
    dataset = read_data(path,200000)
    base,test = sort_data(dataset,0.6)
    g = create_graph(base)
    mg = dgraph_to_mgraph(g)
    nx.write_gpickle(g,prefix+'_graph')
    nx.write_gpickle(mg,prefix+'_mgraph')
    
    outfile = open(prefix+'_raw','ab')
    pickle.dump(dataset,outfile)
    outfile.close()

    outfile = open(prefix+'_test','ab')
    pickle.dump(test,outfile)
    outfile.close()
print('done!')
quit()
'''

#============================== Actual Testing here ==============================#

# This line reads in the pickled data .
# Parameter determines whether it is from a dataset of approx. 50,000 games or 100,000 games
# I just left the smaller dataset since that's what I started with, before I realized that it could get bigger
g1,mg1,t1,g2,mg2,t2,g3,mg3,t3 = read_prepped_data(False)


# IMPORTANT! 
# calculate_fairgoodness() must be called every time before a run of simulation. 
# Or the method will have nothing to read from
# Likewise, stored_pagerank={} must be called every time before a simulation in order to save
# running time (so that repeated random walks do not have to happen)

'''The only part that matters for running the prediction:'''
# simulate(t1,mg1,500,pagerank_easy)
type = 'Common Neighbors'
fn = common_neighbors
thres = 1

# fg = calculate_fairgoodness(mg1,thres)
print('======================')
print(type,'for dataset 1')
stored_pagerank = {}
simulate(t1,g1,thres,fn)

# fg = calculate_fairgoodness(mg2,thres)

print('======================')
print(type,'for dataset 2')
stored_pagerank = {}
simulate(t2,g2,thres,fn)

# fg = calculate_fairgoodness(mg3,thres)

print('======================')
print(type,'for dataset 3')
stored_pagerank = {}
simulate(t3,g3,thres,fn)
