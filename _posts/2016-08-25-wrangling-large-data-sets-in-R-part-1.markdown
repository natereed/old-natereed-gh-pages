---
layout: post
title:  "Working with Large Data Sets in R"
date:   2016-08-22 20:17:27 -0500
categories: R data-wrangling
---

Typically, analysis in R is performed in RAM on a single node compute cluster. 

As the scale of data grows, the analyst is forced to make some tradeoffs among considerations such as time, space and accuracy. A prediction method might be very accurate but unsuitable for production due to its slow running time. A larger training set leads to greater accuracy but tests the limits of hardware.

For very large data sets, R might be in the wrong class of tools altogether. For "medium size" data -- too large to make sense of in a spreadsheet, but too small to justify Hadoop or other "Big Data" solutions -- we would like to use R to develop an effective prototype within the time and space constraints that we inevitably face. 

There are at least a couple ways that we can use to make these tough jobs in R easier:

- Chunking and checkpoints
- Parallelism

The concept of parallelism is well-understood, and there are many resources that explain packages like *foreach* and *parallel*.

Another important concept is how large tasks can be broken up into small pieces, to make it easier to track work. In addition, chunking large tasks makes it possible to recover from failures.  

In a follow-up post, I will show my "chunking" module, which can be used to break up a large piece of work into smaller ones.





