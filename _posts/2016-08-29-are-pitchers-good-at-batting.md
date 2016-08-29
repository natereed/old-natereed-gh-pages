---
layout: page
title: Are Pitchers Good at Batting?
permalink: /are-pitchers-good-at-batting/
---

## Introduction

I was curious about the batting skills of pitchers (and the pitching skills of batters ), so I trawled through [Lahman’s baseball database][lahmans-baseball-database], going back to the beginning of the National League in 1871. I analyzed batting and pitching data to answer the question: “Are Pitchers Good at Batting?” 

Having some limited knowledge of the game, I already had an idea of the answer in the back of my mind. I wanted to check if my preconceptions were correct.

## Background

Pitcher is the [second most highly compensated position][baseball-compensation] next to First Baseman.

Designated Hitter is at the bottom. This surprised me, as I had assumed that great hitters would be very valuable. The viewing public loves hits and home runs, just as they are entertained by touchdowns and goals in football and soccer. Great hitters are responsible for some of the greatest moments in baseball history.

The designated hitter rule, adopted in 1973 by the American League, allows one player to bat in place of the pitcher. This rationale for this rule is the observation that pitchers tend to be weak hitters. Perhaps as expected, when the designated hitter rule was introduced in American League, batting averages trended up. Meanwhile, averages changed little in the National League:

![Batting Averages Time Series](/images/batting-averages-time-series.png)

Based on compensation, it is clear that a combination of fielding and batting skills are more highly valued highly in baseball than batting alone, despite the addition of a batting specialist position in the American League.

In most of history (and to this day in leagues like the National League), pitchers also bat. Other position players may also pitch. Such [versatility has become increasingly common][position-players-increasingly-called-upon-to-pitch] recently. 

Due to circumstances, many players have pitched in relief, and some who are known for their batting skills have pitched well enough to earn multiple trips to the mound. An even smaller group has excelled at both pitching and hitting. Babe Ruth comes to mind as the rare unicorn who was great at both.

## Methodology

I divided the batting data into two groups: pitchers who bat, and all other players. Initially, this selected a large number of players who have done a little of both. The non-pitchers were better batters than the pitchers by a wide margin (0.23 vs. 0.17).

I wanted to consider only players with significant game experience with batting and pitching. I eliminated all players who had fewer than 100 “At Bats”, and from the pitching group I eliminated those with fewer than 100 innings.

![Batting Averages for Pitchers and Non-Pitchers](/images/batting-averages-pitchers-vs-others.png)

Even after filtering out the dilletantes, the pitching group still appears to have a lower average than the non-pitchers (.218 vs. .256).

I used a t-test to test our null hypothesis:

![Batting Averages T-Test](/images/batting-averages-t-test.png)

 The p-value is very low, so we can reject the null hypothesis at the 95% confidence level. 

## Other Observations and Thoughts

The idea that better players get more opportunities may seem self-evident, but it's worth repeating. As we raise the bar for players in our sample sets, by requiring greater numbers of "At Bats," our pitcher group becomes more like the group of other players. These are players that get more opportunities to bat and pitch because they are good enough at each skill to earn a place on the starting roster. Therefore, in such a comparison, we would expect their averages to diverge less than the averages in our test above.

Also, note that the set of pitchers that bat is small compared to the set of other players that bat. As we raise the threshold for "At Bats", we will find fewer and fewer pitchers that bat, to the point where we might not be able to find a large enough sample to come to a conclusion with any degree of certainty.

## Conclusion

For the given data and criteria we used to define these groups, pitchers are clearly worse at batting than other players. The designated hitter rule generated higher batting averages because pitchers, generally speaking, are not good batters.

The iPython notebook for generating the above plots can be found on [github](http://github.com/natereed/springboard-baseball-story.git).

[lahmans-baseball-database]: http://www.seanlahman.com/baseball-archive/statistics/
[baseball-compensation]: http://www.businessinsider.com/major-league-baseballs-highest-paid-positions-sports-chart-of-the-day-2012-9
[position-players-increasingly-called-upon-to-pitch]: http://m.mlb.com/news/article/78938922/position-players-increasingly-called-upon-to-pitch/
