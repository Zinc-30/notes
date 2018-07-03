

  * [**introduction**](#introduction)
  * [**study**](#study)
  * [**theory**](#theory)
  * [**frequentist statistics**](#frequentist-statistics)
  * [**bayesian statistics**](#bayesian-statistics)
  * [**interesting quotes**](#interesting-quotes)
  * [**interesting papers**](#interesting-papers)



---
### introduction

  ["What is statistics?"](http://blogs.sas.com/content/iml/2014/08/05/stiglers-seven-pillars-of-statistical-wisdom.html) by Stephen Stigler  

  "*Aggregation*: It sounds like an oxymoron that you can gain knowledge by discarding information, yet that is what happens when you replace a long list of numbers by a sum or mean. Every day the news media reports a summary of billions of stock market transactions by reporting a single a weighted average of stock prices: the Dow Jones Industrial Average. Statisticians aggregate, and policy makers and business leaders use these aggregated values to make complex decisions.

  *The law of diminishing information*: If 10 pieces of data are good, are 20 pieces twice as good? No, the value of additional information diminishes like the square root of the number of observations. The square root appears in formulas such as the standard error of the mean, which describes the probability that the mean of a sample will be close to the mean of a population.

  *Likelihood*: Some people say that statistics is "the science of uncertainty." One of the pillars of statistics is being able to confidently state how good a statistical estimate is. Hypothesis tests and p-values are examples of how statisticians use probability to carry out statistical inference.

  *Intercomparisons*: When analyzing data, statisticians usually make comparisons that are based on differences among the data. This is different than in some fields, where comparisons are made against some ideal "gold standard." Well-known analyses such as ANOVA and t-tests utilize this pillar.

  *Regression and multivariate analysis*: Children that are born to two extraordinarily tall parents tend to be shorter than their parents. Similarly, if both parents are shorter than average, the children tend to be taller than the parents. This is known as regression to the mean. Regression is the best known example of multivariate analysis, which also includes dimension-reduction techniques and latent factor models.

  *Design*: A pillar of statistics is the design of experiments, and—by extension—all data collection and planning that leads to good data. Included in this pillar is the idea that random assignment of subjects to design cells improves the analysis. This pillar is the basis for agricultural experiments and clinical trials, just to name two examples.

  *Models and Residuals*: This pillar enables you to examine shortcomings of a model by examining the difference between the observed data and the model. If the residuals have a systematic pattern, you can revise your model to explain the data better. You can continue this process until the residuals show no pattern. This pillar is used by statistical practitioners every time that they look at a diagnostic residual plot for a regression model."

  *(Stephen Stigler)*

----

  "On Computational Thinking, Inferential Thinking and Data Science" by Michael I. Jordan
	([first talk](https://youtube.com/watch?v=bIfB1fj8xGQ) `video`, [second talk](https://youtube.com/watch?v=cUQ5yYr8JuI) `video`)  

  "The rapid growth in the size and scope of datasets in science and technology has created a need for novel foundational perspectives on data analysis that blend the inferential and computational sciences. That classical perspectives from these fields are not adequate to address emerging problems in "Big Data" is apparent from their sharply divergent nature at an elementary level-in computer science, the growth of the number of data points is a source of "complexity" that must be tamed via algorithms or hardware, whereas in statistics, the growth of the number of data points is a source of "simplicity" in that inferences are generally stronger and asymptotic results can be invoked. On a formal level, the gap is made evident by the lack of a role for computational concepts such as "runtime" in core statistical theory and the lack of a role for statistical concepts such as "risk" in core computational theory."

  *(Michael I. Jordan)*

  Michael I. Jordan on [statistics in machine learning](https://youtube.com/watch?v=uyZOcUDhIbY&t=17m27s) `video`  
  Michael I. Jordan on [theory in machine learning](https://youtube.com/watch?v=uyZOcUDhIbY&t=23m1s) `video`  

----

  ["Are Machine Learning and Statistics Complementary"](https://www.ics.uci.edu/~welling/publications/papers/WhyMLneedsStatistics.pdf) by Max Welling



---
### study

  ["The Probability and Statistics Cookbook"](https://github.com/mavam/stat-cookbook/releases/download/0.2.4/stat-cookbook.pdf) by Matthias Vallentin  

  ["Probability and Statistics for Data Science"](http://cims.nyu.edu/~cfgranda/pages/stuff/probability_stats_for_DS.pdf) by Carlos Fernandez-Granda  
  ["Computer Age Statistical Inference"](https://web.stanford.edu/~hastie/CASI_files/PDF/casi.pdf) by Bradley Efron and Trevor Hastie  
  ["Advanced Data Analysis from an Elementary Point of View"](http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/ADAfaEPoV.pdf) by Cosma Shalizi  

  ["All of Statistics: A Concise Course in Statistical Inference"](http://read.pudn.com/downloads158/ebook/702714/Larry%20Wasserman_ALL%20OF%20Statistics.pdf) by Larry Wasserman  

----

  [course](https://youtube.com/playlist?list=PLCzY7wK5FzzPANgnZq5pIT3FOomCT1s36) by Joe Blitzstein `video`  

  [course](https://stepik.org/course/326/) from Computer Science Center `video` `in russian` ([videos](https://youtube.com/playlist?list=PLwwk4BHih4fgFBDweG6t8DDPMzaWIbGXr))  
  [course](https://compscicenter.ru/courses/math-stat/2015-spring/) from Computer Science Center `video` `in russian`  
  [course](https://compscicenter.ru/courses/math-stat/2013-spring/) from Computer Science Center `video` `in russian`  

  [course](https://coursera.org/specializations/machine-learning-data-analysis) from Yandex `video` `in russian`  
  [course](https://youtube.com/playlist?list=PLrCZzMib1e9p5F99rIOzugNgQP5KHHfK8) from Mail.ru `video` `in russian`  
  [course](https://coursera.org/learn/ekonometrika/) from HSE `video` `in russian`  



---
### applications

  ["A/B Testing at Scale: Accelerating Software Innovation"](http://exp-platform.com/2017abtestingtutorial/) tutorial `video`  
  ["A/B Testing at Scale: Accelerating Software Innovation"](https://youtube.com/watch?v=RoJ_s7Bb4qM) short tutorial `video`  

  ["A Dirty Dozen: Twelve Common Metric Interpretation Pitfalls in Online Controlled Experiments"](http://kdd.org/kdd2017/papers/view/a-dirty-dozen-twelve-common-metric-interpretation-pitfalls-in-online-contro) by Gupta et al. `paper` ([talk](https://youtube.com/watch?v=nFfO9bOrp88) `video`)

  ["Online Controlled Experiments"](https://youtube.com/watch?v=qyW8Rx0mx3o) by Ron Kohavi `video`



---
### theory

  [4 views of statistics: Frequentist, Bayesian, Likelihood, Information-Theoretic](http://labstats.net/articles/overview.html)

----

  "In essence, Bayesian means probabilistic. The specific term exists because there are two approaches to probability. Bayesians think of it as a measure of belief, so that probability is subjective and refers to the future. Frequentists have a different view: they use probability to refer to past events - in this way it’s objective and doesn’t depend on one’s beliefs."


  http://allendowney.blogspot.ru/2016/09/bayess-theorem-is-not-optional.html

  difference between bayesian and frequentist expected loss - https://en.wikipedia.org/wiki/Loss_function#Expected_loss
  example of frequentist (p-values) and bayesian (bayes factor) statistical hypothesis testing - https://en.wikipedia.org/wiki/Bayes_factor#Example


  http://blog.efpsa.org/2014/11/17/bayesian-statistics-what-is-it-and-why-do-we-need-it-2/
  http://blog.efpsa.org/2015/08/03/bayesian-statistics-why-and-how/


  advantages of bayesian inference - http://bayesian-inference.com/advantagesbayesian
  advantages of frequentist inference - http://bayesian-inference.com/advantagesfrequentist
  frequentist vs bayesian vs machine learning - http://stats.stackexchange.com/a/73180

  "Frequentism and Bayesianism: A Python-driven Primer" - http://arxiv.org/abs/1411.5018

  http://nowozin.net/sebastian/blog/becoming-a-bayesian-part-1.html
  http://nowozin.net/sebastian/blog/becoming-a-bayesian-part-2.html
  http://nowozin.net/sebastian/blog/becoming-a-bayesian-part-3.html


  http://jakevdp.github.io/blog/2014/03/11/frequentism-and-bayesianism-a-practical-intro/
  http://jakevdp.github.io/blog/2014/06/06/frequentism-and-bayesianism-2-when-results-differ/
  http://jakevdp.github.io/blog/2014/06/12/frequentism-and-bayesianism-3-confidence-credibility/


  "Bayesian or Frequentist, Which Are You?" by Michael I. Jordan - http://videolectures.net/mlss09uk_jordan_bfway/

----

  "Model-based statistics assumes that the observed data has been produced from a random distribution or probability model. The model usually involves some unknown parameters. Statistical inference aims to learn the parameters from the data. This might be an end in itself - if the parameters have interesting real world implications we wish to learn - or as part of a larger workflow such as prediction or decision making. Classical approaches to statistical inference are based on the probability (or probability density) of the observed data y0 given particular parameter values θ. This is known as the likelihood function, π(y0|θ). Since y0 is fixed this is a function of θ and so can be written L(θ). Approaches to inference involve optimising this, used in maximum likelihood methods, or exploring it, used in Bayesian methods.  
  A crucial implicit assumption of both approaches is that it’s possible and computationally inexpensive to numerically evaluate the likelihood function. As computing power has increased over the last few decades, there are an increasing number of interesting situations for which this assumption doesn’t hold. Instead models are available from which data can be simulated, but where the likelihood function is intractable, in that it cannot be numerically evaluated in a practical time.  
  In Bayesian approach to inference a probability distribution must be specified on the unknown parameters, usually through a density π(θ). This represents prior beliefs about the parameters before any data is observed. The aim is to learn the posterior beliefs resulting from updating the prior to incorporate the observations. Mathematically this is an application of conditional probability using Bayes theorem: the posterior is π(θ|y0)=kπ(θ)L(θ), where k is a constant of proportionality that is typically hard to calculate. A central aim of Bayesian inference is to produce methods which approximate useful properties of the posterior in a reasonable time."

----

  "The traditional ‘frequentist’ methods which use only sampling distributions are usable and useful in many particularly simple, idealized problems; however, they represent the most proscribed special cases of probability theory, because they presuppose conditions (independent repetitions of a ‘random experiment’ but no relevant prior information) that are hardly ever met in real problems. This approach is quite inadequate for the current needs of science. In addition, frequentist methods provide no technical means to eliminate nuisance parameters or to take prior information into account, no way even to use all the information in the data when sufficient or ancillary statistics do not exist. Lacking the necessary theoretical principles, they force one to ‘choose a statistic’ from intuition rather than from probability theory, and then to invent ad hoc devices (such as unbiased estimators, confidence intervals, tail-area significance tests) not contained in the rules of probability theory. Each of these is usable within the small domain for which it was invented but, as Cox’s theorems guarantee, such arbitrary devices always generate inconsistencies or absurd results when applied to extreme cases."

  "All of these defects are corrected by use of Bayesian methods, which are adequate for what we might call ‘well-developed’ problems of inference. As Jeffreys demonstrated, they have a superb analytical apparatus, able to deal effortlessly with the technical problems on which frequentist methods fail. They determine the optimal estimators and algorithms automatically, while taking into account prior information and making proper allowance for nuisance parameters, and, being exact, they do not break down – but continue to yield reasonable results – in extreme cases. Therefore they enable us to solve problems of far greater complexity than can be discussed at all in frequentist terms. All this capability is contained already in the simple product and sum rules of probability theory interpreted as extended logic, with no need for – indeed, no room for – any ad hoc devices."

  "Before Bayesian methods can be used, a problem must be developed beyond the ‘exploratory phase’ to the point where it has enough structure to determine all the needed apparatus (a model, sample space, hypothesis space, prior probabilities, sampling distribution). Almost all scientific problems pass through an initial exploratory phase in which we have need for inference, but the frequentist assumptions are invalid and the Bayesian apparatus is not yet available. Indeed, some of them never evolve out of the exploratory phase. Problems at this level call for more primitive means of assigning probabilities directly out of our incomplete information. For this purpose, the Principle of maximum entropy has at present the clearest theoretical justification and is the most highly developed computationally, with an analytical apparatus as powerful and versatile as the Bayesian one. To apply it we must define a sample space, but do not need any model or sampling distribution. In effect, entropy maximization creates a model for us out of our data, which proves to be optimal by so many different criteria that it is hard to imagine circumstances where one would not want to use it in a problem where we have a sample space but no model."

  *(E. T. Jaynes)*

----

  "The frequentist vs. Bayesian debate that raged for decades in statistics before sputtering out in the 90s had more of a philosophical flavor. Starting with Fisher, frequentists argued that unless a priori probabilities were known exactly, they should not be "guessed" or "intuited", and they created many tools that did not require the specification of a prior. Starting with Laplace, Bayesians quantified lack of information by means of a "uninformative" or "objective" uniform prior, using Bayes theorem to update their information as more data came in. Once it became clear that this uniform prior was not invariant under transformation, Bayesian methods fell out of mainstream use. Jeffreys led a Bayesian renaissance with his invariant prior, and Lindley and Savage poked holes in frequentist theory. Statisticians realized that things weren't quite so black and white, and the rise of MCMC methods and computational statistics made Bayesian inference feasible in many, many new domains of science. Nowadays, few statisticians balk at priors, and the two strands have effectively merged (consider the popularity of empirical Bayes methods, which combine the best of both schools). There are still some Bayesians that consider Bayes theorem the be-all-end-all approach to inference, and will criticize model selection and posterior predictive checks on philosophical grounds. However, the vast majority of statisticians will use whatever method is appropriate. The problem is that many scientists aren't yet aware of/trained in Bayesian methods and will use null hypothesis testing and p-values as if they're still the gold standard in statistics."



---
### frequentist statistics

  "Informally, a p-value is the probability under a specified statistical model that a statistical summary of the data (for example, the sample mean difference between two compared groups) would be equal to or more extreme than its observed value."

  "P values are commonly used to test (and dismiss) a ‘null hypothesis’, which generally states that there is no difference between two groups, or that there is no correlation between a pair of characteristics. The smaller the P value, the less likely an observed set of values would occur by chance — assuming that the null hypothesis is true. A P value of 0.05 or less is generally taken to mean that a finding is statistically significant and warrants publication."


  http://allendowney.blogspot.ru/2011/05/there-is-only-one-test.html  
  http://allendowney.blogspot.ru/2011/06/more-hypotheses-less-trivia.html  


  "The ASA's statement on p-values: context, process, and purpose" [http://dx.doi.org/10.1080/00031305.2016.1154108]:

  - P-values can indicate how incompatible the data are with a specified statistical model.

	"A p-value provides one approach to summarizing the incompatibility between a particular set of data and a proposed model for the data. The most common context is a model, constructed under a set of assumptions, together with a so-called “null hypothesis.” Often the null hypothesis postulates the absence of an effect, such as no difference between two groups, or the absence of a relationship between a factor and an outcome. The smaller the p-value, the greater the statistical incompatibility of the data with the null hypothesis, if the underlying assumptions used to calculate the p-value hold. This incompatibility can be interpreted as casting doubt on or providing evidence against the null hypothesis or the underlying assumptions."

  - P-values do not measure the probability that the studied hypothesis is true, or the probability that the data were produced by random chance alone.

	"Researchers often wish to turn a p-value into a statement about the truth of a null hypothesis, or about the probability that random chance produced the observed data. The p-value is neither. It is a statement about data in relation to a specified hypothetical explanation, and is not a statement about the explanation itself."

  - Scientific conclusions and business or policy decisions should not be based only on whether a p-value passes a specific threshold.

	"Practices that reduce data analysis or scientific inference to mechanical “bright-line” rules (such as “p < 0.05”) for justifying scientific claims or conclusions can lead to erroneous beliefs and poor decision-making. A conclusion does not immediately become “true” on one side of the divide and “false” on the other. Researchers should bring many contextual factors into play to derive scientific inferences, including the design of a study, the quality of the measurements, the external evidence for the phenomenon under study, and the validity of assumptions that underlie the data analysis. Pragmatic considerations often require binary, “yes-no” decisions, but this does not mean that p-values alone can ensure that a decision is correct or incorrect. The widespread use of “statistical significance” (generally interpreted as “p ≤ 0.05”) as a license for making a claim of a scientific finding (or implied truth) leads to considerable distortion of the scientific process."

  - Proper inference requires full reporting and transparency.

	"P-values and related analyses should not be reported selectively. Conducting multiple analyses of the data and reporting only those with certain p-values (typically those passing a significance threshold) renders the reported p-values essentially uninterpretable. Cherry-picking promising findings, also known by such terms as data dredging, significance chasing, significance questing, selective inference and “p-hacking”, leads to a spurious excess of statistically significant results in the published literature and should be vigorously avoided. One need not formally carry out multiple statistical tests for this problem to arise: Whenever a researcher chooses what to present based on statistical results, valid interpretation of those results is severely compromised if the reader is not informed of the choice and its basis. Researchers should disclose the number of hypotheses explored during the study, all data collection decisions, all statistical analyses conducted and all p-values computed. Valid scientific conclusions based on p-values and related statistics cannot be drawn without at least knowing how many and which analyses were conducted, and how those analyses (including p-values) were selected for reporting."

  - A p-value, or statistical significance, does not measure the size of an effect or the importance of a result.

	"Statistical significance is not equivalent to scientific, human, or economic significance. Smaller p-values do not necessarily imply the presence of larger or more important effects, and larger p-values do not imply a lack of importance or even lack of effect. Any effect, no matter how tiny, can produce a small p-value if the sample size or measurement precision is high enough, and large effects may produce unimpressive p-values if the sample size is small or measurements are imprecise. Similarly, identical estimated effects will have different p-values if the precision of the estimates differs."

  - By itself, a p-value does not provide a good measure of evidence regarding a model or hypothesis.

	"Researchers should recognize that a p-value without context or other evidence provides limited information. For example, a p-value near 0.05 taken by itself offers only weak evidence against the null hypothesis. Likewise, a relatively large p-value does not imply evidence in favor of the null hypothesis; many other hypotheses may be equally or more consistent with the observed data. For these reasons, data analysis should not end with the calculation of a p-value when other approaches are appropriate and feasible."


  "
  - The p-value doesn't tell scientists what they want (it is the probability of the data given that H0 is true, and scientists would like the probability of H0 or H1 given the data)
  - H0 is often known to be false
  - P-values are widely misunderstood
  - Leads to binary yes/no thinking
  - Prior information is never taken into account (Bayesian argument)
  - A small p-value could reflect a very large sample size rather than a meaningful difference
  - Leads to publication bias, because significant results (i.e. p < 0.05) are more likely to be published
  "


  "Null Hypothesis Significance Testing Never Worked" - http://fharrell.com/2017/01/null-hypothesis-significance-testing.html  
  http://quillette.com/2015/11/13/the-great-statistical-schism/  



---
### bayesian statistics

  http://blog.efpsa.org/2014/11/17/bayesian-statistics-what-is-it-and-why-do-we-need-it-2/  
  http://blog.efpsa.org/2015/08/03/bayesian-statistics-why-and-how/  

  Kruschke - "Doing Bayesian Data Analysis" - http://www.users.csbsju.edu/~mgass/robert.pdf

  "Statistical Computing for Scientists and Engineers" course by Nicholas Zabaras - https://zabaras.com/statisticalcomputing


  https://alexanderetz.com/understanding-bayes/ :
  - a look at the likelihood - https://alexanderetz.com/2015/04/15/understanding-bayes-a-look-at-the-likelihood/
  - updating priors via the likelihood
  - evidence vs. conclusions
  - what is the maximum Bayes factor for a given p value
  - posterior probabilities vs. posterior odds
  - objective vs. subjective Bayes
  - prior probabilities for models vs. parameters
  - strength of evidence vs. probability of obtaining that evidence
  - the Jeffreys-Lindley paradox
  - when do Bayesians and frequentists agree and why?
  - bayesian model averaging
  - bayesian bias mitigation
  - bayesian updating over multiple studies
  - does Bayes have error control?


  "A likelihood (of data given model parameters) is similar to a probability, but the area under a likelihood curve does not add up to one like it does for a probability density. It treats the data as fixed (rather than as a random variable) and the likelihood of two different models can be compared by taking the ratio of their likelihoods, and a test of signficance can be performed."

  "Likelihood is not a probability, but it is proportional to a probability. The likelihood of a hypothesis (H) given some data (D) is proportional to the probability of obtaining D given that H is true, multiplied by an arbitrary positive constant (K). In other words, L(H|D) = K · P(D|H). Since a likelihood isn’t actually a probability it doesn’t obey various rules of probability. For example, likelihood need not sum to 1.

  A critical difference between probability and likelihood is in the interpretation of what is fixed and what can vary. In the case of a conditional probability, P(D|H), the hypothesis is fixed and the data are free to vary. Likelihood, however, is the opposite. The likelihood of a hypothesis, L(H|D), conditions on the data as if they are fixed while allowing the hypotheses to vary."

  Likelihood:
  - is central to almost all of statistics
  - treats the data as fixed (once the experiment is complete, the data are fixed)
  - allows one to compare hypotheses given the data
  - captures the evidence in the data
  - likelihoods can be easily combined, for example from two independent studies
  - prior information can easily included (Bayesian analysis)
  - seems to be the way we normally think (Pernerger & Courvoisier, 2010)

  Bayes factors:
  - not biased against H0
  - allow us to state evidence for the absence of an effect
  - condition only on the observed data
  - allow to stop experiment once the data is informative enough
  - subjective just as p-values

  "Because complex models can capture many different observations, their prior on parameters p(θ) is spread out wider than those of simpler models. Thus there is little density at any specific point - because complex models can capture so many data points; taken individually, each data point is comparatively less likely. For the marginal likelihood, this means that the likelihood gets multiplied with these low density values of the prior, which decreases the overall marginal likelihood. Thus model comparison via Bayes factors incorporates an automatic Ockham’s razor, guarding us against overfitting. While classical approaches like the AIC naively add a penalty term (2 times the number of parameters) to incorporate model complexity, Bayes factors offer a more natural and principled approach to this problem."


  pathologies of frequentist statistics according to E. T. Jaynes - https://youtube.com/watch?v=zZkwzvrO-pU



---
### interesting quotes

  (Andrew Gelman) "In reality, null hypotheses are nearly always false. Is drug A identically effective as drug B? Certainly not. You know before doing an experiment that there must be some difference that would show up given enough data."

  (Jim Berger) "A small p-value means the data were unlikely under the null hypothesis. Maybe the data were just as unlikely under the alternative hypothesis. Comparisons of hypotheses should be conditional on the data.

  (Stephen Ziliak, Deirdra McCloskey) "Statistical significance is not the same as scientific significance. The most important question for science is the size of an effect, not whether the effect exists."

  (William Gosset) "Statistical error is only one component of real error, maybe a small component. When you actually conduct multiple experiments rather than speculate about hypothetical experiments, the variability of your data goes up."

  (John Ioannidis) "Small p-values do not mean small probability of being wrong. In one review, 74% of studies with p-value 0.05 were found to be wrong."

  () "Empirical model comparisons based on real data compare (model1, estimator1, algorithm1) with (model2, estimator2, algorithm2). Saying my observations, xᵢ, are IID draws from a Bernoulli(p) random variable is a model for my data. Using the sample mean, p̂ = mean(xᵢ), to estimate the value of p is an estimator for that model. Computing the sample mean as sum(x)/length(x) is an algorithm if you assume sum and length are primitives. The distinctions matter because you always fit models to data using a triple of (model, estimator, algorithm)."

  () "Data Science is a lot more than machine learning:  
  - understanding goals (sometimes requires background research)  
  - how to get the right data  
  - figuring out what to measure or optimize  
  - beware the lazy path of computing what's easy but wrong"  



---
### interesting papers

  Breiman - "Statistical Modeling: The Two Cultures"  
  Norvig - "Warning Signs in Experimental Design and Interpretation" [http://norvig.com/experiment-design.html]  
  Debrouwere, Goetghebeur - "The Statistical Crisis in Science" [http://lib.ugent.be/fulltxt/RUG01/002/304/385/RUG01-002304385_2016_0001_AC.pdf]  
  Ioannidis - "Why most published research findings are false" [http://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0020124]  
  Goodman, Greenland - "Why Most Published Research Findings Are False: Problems in the Analysis"  
  Ioannidis - "Why Most Published Research Findings Are False: Author's Reply to Goodman and Greenland"  
  Moonesinghe Khoury, Janssens - "Most published research findings are false - But a little replication goes a long way"  
  Leek, Jager - "Is most published research really false?"  
  Aitchison, Corradi, Latham - "Zipf’s Law Arises Naturally When There Are Underlying, Unobserved Variables" [http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005110]  
