I uploaded the pickled versions of the data I had, but I'm not sure if it's going to work the same in other systems.
If they do not work, the data can be found ![here](https://www.ficsgames.org/download.html)

Packages needed to run this code:
- tqdm (for status bar when testing the predictions)        
- python-chess (for unpacking/parsing the game data from .pgn files)

Changes from the update presentation:
Looks like I lied about 70 percent being below average for a 'goog' prediction rate. I checked my references again (and also tried actually randomly guessing), and it seems that around 50% is average, and nearing 70% is actually a very good predictor.

And for some reason, pagerank is actually not as good as common neighbors or edge weights(win rates) method. 
So far, a 50/50 (coinflip) perdiction yields ~48% accuracy.

For reference, this is the performance of the coinflip method
|Prediction Method|Dataset 1|Dataset 2|Dataset 3|
|-|-|-|-|
|Coin Flip (50/50)|48.1%|48.05%|47.71%|

### Performances so far
#### Threshold = 1
|Prediction Method|Dataset 1|Dataset 2|Dataset 3|
|-|-|-|-|
|Common Neighbors|56.73%|56.23%|56.42%|
|Edge Weights|57.95%|57.62%|57.81%|

(threshold of 1 means that it will not skip prediction for that match if there is at least one past record of each player, otherwise it does)

Also, increasing threshold of past records for comm-neighbors and edge weights actually brings down the accuracy, which I assume is from the lack of samples. So I was also mistaken about this in my prior presentation as well.

### Performance of number paths method:
#### Path length is also including the players being evaluated
|Path Length threshold|Dataset 1|Dataset 2|Dataset 3|
|:-:|-|-|-|
|3|55.98%|54.73%|55.32%|
|4|58.45%|57.66%|57.72%|


**Even at threshold of length 4, each run took over 10 minutes, so it's pretty inefficient, compared to other methods**

For the pagerank algorithm, I changed my implementation a bit so that it would work better for my example (or so I think)
And for a number of given iterations, the algorithm starts with a given source. It will try to visit other nodes that a directed edge takes it until it meets a dead end or the random value generated is less than alpha (currently set to 0.1, this is just to give some more randomness and finish the walks early so it doesn't take forever) 
So it is really a derivative of the 'personalized pagerank' we did in class.
It will do this for a given number of iterations, and each time I evaluate a matchup, I run this for both players, which is probably why this is so slow. In order to avoid doing the calculation for the same player every time, I am storing the for each player every time I calculate it, so that I can re-use it for recurring players.

My Implementation of a random walk:
```
def random_walk(source,g,num):
    visits = {}
    alpha = 0.1
    for v in g.nodes(): visits[v] = 0
    for i in range(num):

        # we want to use edges because calling 
        # successors returns the multi-edged ones only
        # once, but the edges show all the edges
    
        d = source
        edges = list(g.edges(source))
        while random.random() > alpha and edges:

            # stops either when the random 
            # value is lower than alpha
            # or when there are no more edges to take

            d = random.choice(edges)[1]
            edges = list(g.edges(d))
        visits[d] += 1
    for v in g.nodes(): visits[v] /= num
    return visits
```

### My Pagerank Implementation Performance
 **Since the output for this tends to be more random, I will run these guys multiple times**

##### Trial 1
|Number of 'Random Walk'|Dataset 1|Dataset 2|Dataset 3|
|-|-|-|-|
|100|48.42%|50.45%|51.12%|
|1000|49.84%|50.42%|51.27%|
|10000|50.26%|50.8%|51.49%|

##### Trial 2
|Number of 'Random Walk'|Dataset 1|Dataset 2|Dataset 3|
|-|-|-|-|
|100|48.37%|49.41%|50.45%|
|1000|49.98%|49.98%|51.1%|
|10000|50.5%|50.98%|51.4%|

##### Trial 3
|Number of 'Random Walk'|Dataset 1|Dataset 2|Dataset 3|
|-|-|-|-|
|100|49.15%|48.63%|49.97%|
|1000|50.18%|50.96%|51.1%|
|10000|50.08%|50.97%|51.51%|

**at iteration=10000, it was taking like an hour or so to complete the tests (so 40000 iterations of future matches), so I'm not sure if it is worth trying anything bigger, although the accuracy does go up as the iteration does**