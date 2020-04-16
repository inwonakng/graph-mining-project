I uploaded the pickled versions of the data I had, but I'm not sure if it's going to work the same in other systems.
If they do not work, the data can be found ![here](https://www.ficsgames.org/download.html)

Packages needed to run this code:
tqdm, python-chess

Changes from the update presentation:
Looks like I lied about 70 percent being below average for a 'goog' prediction rate. I checked my references again (and also tried actually randomly guessing), and it seems that around 50% is average, and nearing 70% is actually a very good predictor.

And for some reason, pagerank is actually not as good as common neighbors or edge weights(win rates) method. 
So far, a 50/50 (coinflip) perdiction yields ~48% accuracy.
Edge weights (threshold of 1 means that it will not skip prediction for that match if there is at least one past record of each player, otherwise it does):

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

Also, increasing threshold of past records for comm-neighbors and edge weights actually brings down the accuracy, which I assume is from the lack of samples. So I was also mistaken about this in my prior presentation as well.

### Performance of number paths method:
#### Path length is also including the players being evaluated
|Path Length threshold|Dataset 1|Dataset 2|Dataset 3|
|-|-|-|-|
|3|55.98%|54.73%|55.32%|
|4|%|%|%|

**Even at threshold of length 4, each run took over 10 minutes, so it's pretty inefficient, compared to toerh methods**

For the pagerank algorithm, I changed my implementation a bit so that it would work better for my example (or so I thought)
And for a number of given iterations, it 

 
