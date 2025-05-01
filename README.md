# imc-prosperity-3
 
These were our submissions for IMC Prosperity 3, a trading competition hosted by IMC. Team members included [Michael Lee](https://www.linkedin.com/in/ml64/), [Buck Arney](https://www.linkedin.com/in/buck-arney/), and Akash Deo. This was the first trading competition for all three of our members.

Submissions for all rounds except for algorithmic trading for round one are attached in their respective folders. The submission for Round 1 is almost identical to the submission for Round 2, absent the inclusion of the new products introduced in Round 2.

The "Michael work" folder includes data analysis and code fragments written by Michael.

## Round 1
Unfortunately, Buck and Akash were at the National Debate Tournament during the first round, so we didn't have a ton of time to work on this stage.

This stage included three products: RAINFOREST_RESIN, KELP, and SQUID_INK. The hints that were given is that RAINFOREST_RESIN hangs at about 10K seashells (the main currency) and that the price of SQUID_INK reverts to its mean in the short run.

Our original attempt for rainforest resin was to simply buy any order below 10K seashells and to sell to any order above 10K seashells. However, we found that this constantly put us at our position limits, meaning that if a more profitable opportunity arrived later, we wouldn't be able to capitalize on it.

Our solution was to dynamically limit the amount of resin we could buy or sell, depending on how much the price deviated from 10K; if the best buy price was only at 10001, for instance, we would only sell up to 40% of our position limit; that way, if someone offered to buy at 10003 in the next timestamp, we would still be able to sell.

Kelp and ink were more tricky to trade. Based on the data for day -2, the price of kelp and ink seemed to converge. As a result, I tried to test a pair trading strategy with kelp and ink; however, this pattern didn't hold up for other days so it was scrapped. 

I didn't have a lot of time to write kelp trading, so I ended up just taking the VWAP at each state as the "true price," selling to any orders above and buying any orders below. 

For ink, I calculated the exponential moving average at each time stamp. Then, I calculated the standard deviation of the moving average over time at each time stamp. Lastly, I calculated z-score by subtracting the current EMA from the current mid price, dividing by the standard deviation of moving averages. I do not think this is how we should have been calculating z-score. In any case, if the magnitude of the z-score was greater than 2, we made trades, again dynamically limiting how much we trade based on the divergence between the current mid price and EMA.


The last thing I did for ink was that I set up a system to try to keep our position close to zero. As a result, we would send out profitable orders for the next timestamp that opposed our position. For example, if we had just bought 30 ink at 100 seashells, we would send out an ask order for 30 ink at 102 seashells.

Manual trading was simple. Given a matrix of imperfect exchange rates between four currencies (i.e. the exchange rate between asset 1->asset 2 was not the reciprocal of exchange rate between asset 2->asset 1), the goal was to find a set of five trades that maximized profit. The elegant way to do this is through some dynamic programming solution; however, since it was only five trades, we ended up brute-forcing to get the best solution. 

## Round 2 

This round Buck and Akash were back and able to help with the next challenge, trading a basket of commodities and the specific commodities that make up the basket.  
