

  * [applications](#applications)
  * [study](#study)
  * [challenges](#challenges)
  * [interesting papers](#interesting-papers)



---
### applications

#### item-to-item similarity modeling

  * "Two Decades of Recommender Systems at Amazon.com" (Amazon) [https://www.computer.org/csdl/mags/ic/2017/03/mic2017030012.html]
	- learning item-to-item similarity on offline data (e.g. item2 often bought with item1)

  * exponential family embeddings [https://youtu.be/zwcjJQoK8_Q?t=15m14s] [https://arxiv.org/abs/1608.00778] [https://github.com/mariru/exponential_family_embeddings] [https://github.com/franrruiz/p-emb]
	- identifying substitutes and co-purchases in high-scale consumer data (basket analysis)

  * prod2vec (Yahoo) [https://youtube.com/watch?v=W56fZewflRw] [https://arxiv.org/abs/1606.07154]



#### user preferences modeling

  * "Latent LSTM Allocation" (Amazon) [https://youtube.com/watch?v=ofaPq5aRKZ0] [https://vimeo.com/240608072] [http://proceedings.mlr.press/v70/zaheer17a/zaheer17a.pdf] [https://arxiv.org/abs/1711.11179]

  * "DropoutNet: Addressing Cold Start in Recommender Systems" [https://youtu.be/YSQqHlQwQDY?t=1h44m18s] [https://papers.nips.cc/paper/7081-dropoutnet-addressing-cold-start-in-recommender-systems] [https://github.com/HongleiXie/DropoutNet]

  * "VAE for Collaborative Filtering" (Netflix) [https://youtube.com/watch?v=gRvxr47Gj3k] [https://arxiv.org/abs/1802.05814] [https://github.com/dawenl/vae_cf]

  * "Content-based recommendations with Poisson factorization" [http://www.fields.utoronto.ca/video-archive/2017/03/2267-16706  (26:36)] [http://www.cs.toronto.edu/~lcharlin/papers/GopalanCharlinBlei_nips14.pdf]
	collaborative topic models:
	- blending factorization-based and content-based recommendation
	- describing user preferences with interpretable topics

  * "Scalable Recommendation with Hierarchical Poisson Factorization" [https://youtu.be/zwcjJQoK8_Q?t=41m49s] [http://auai.org/uai2015/proceedings/papers/208.pdf] [https://github.com/premgopalan/hgaprec]
	- discovering correlated preferences (devising new utility models and other factors such as time of day, date, in stock, customer demographic information)

  * user2vec (Yahoo) [https://youtube.com/watch?v=W56fZewflRw] [https://arxiv.org/abs/1606.07154]

  * LightFM [https://youtube.com/watch?v=EgE0DUrYmo8] [https://arxiv.org/abs/1507.08439] [https://github.com/lyst/lightfm/]

  * "Causal Inference for Recommendation" [http://people.hss.caltech.edu/~fde/UAI2016WS/papers/Liang.pdf] [http://people.hss.caltech.edu/~fde/UAI2016WS/talks/Dawen.pdf] [http://www.homepages.ucl.ac.uk/~ucgtrbd/whatif/David.pdf]



#### item features modeling

  * "Wide & Deep Learning" (Google) [https://youtube.com/watch?v=NV1tkZ9Lq48] [https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html] [https://arxiv.org/abs/1606.07792] [https://www.tensorflow.org/tutorials/wide_and_deep]

  * BigARTM (Yandex) [https://youtu.be/3Lxb-DqPtv4?t=1h45m9s] [https://youtube.com/watch?v=eJzNAhsbQNI] [https://youtube.com/watch?v=Y7lGYjJ7TR8] [https://youtube.com/watch?v=00qF2yMuRkQ] [http://www.machinelearning.ru/wiki/images/d/d5/Voron17survey-artm.pdf] [https://github.com/bigartm/bigartm]

  * Spotlight [https://youtube.com/watch?v=ZkBQ6YA9E40] [https://maciejkula.github.io/spotlight/]



#### active learning

  * Microsoft Custom Decision Service [https://youtube.com/watch?v=7ic_d5TeIUk] [https://vimeo.com/240429210] [http://arxiv.org/abs/1606.03966] [https://github.com/Microsoft/mwt-ds]

	https://ds.microsoft.com
	https://azure.microsoft.com/en-us/services/cognitive-services/custom-decision-service/
	https://docs.microsoft.com/en-us/azure/cognitive-services/custom-decision-service/custom-decision-service-overview
	https://blogs.microsoft.com/next/2016/12/01/machine-learning-breakthroughs-abound-researchers-look-democratize-benefits/

  * conversational recommendations [https://youtube.com/watch?v=udrkPBIb8D4] [https://youtube.com/watch?v=nLUfAJqXFUI] [https://chara.cs.illinois.edu/sites/fa16-cs591txt/pdf/Christakopoulou-2016-KDD.pdf]



---
### study

  "Recommender Systems: The Textbook" by Charu Aggarwal - http://charuaggarwal.net/Recommender-Systems.htm [https://yadi.sk/i/eMDtp31h3P8AxL]
  "Recommender Systems Handbook" by Ricci, Rokach, Shapira, Kantor - http://www.cs.ubbcluj.ro/~gabis/DocDiplome/SistemeDeRecomandare/Recommender_systems_handbook.pdf


  tutorial by Xavier Amatriain - http://technocalifornia.blogspot.ru/2014/08/introduction-to-recommender-systems-4.html

  overview by Alex Smola - https://youtube.com/watch?v=gCaOa3W9kM0
  overview by Alex Smola - https://youtube.com/watch?v=xMr7I-OypVY

  "Lessons Learned from Building Real-life Recommender Systems" by Xavier Amatriain and Deepak Agarwal - https://youtube.com/watch?v=VJOtr47V0eo
  "The Recommender Problem Revisited" by Xavier Amatriain - http://videolectures.net/kdd2014_amatriain_mobasher_recommender_problem/


  overview by Michael Rozner (in russian) - https://youtube.com/watch?v=Us4KJkJiYrM
  overview by Sergey Nikolenko (in russian) - https://youtube.com/watch?v=mr8u54jsveA + https://youtube.com/watch?v=cD47Ssp_Flk + https://youtube.com/watch?v=OFyb8ukrRDo
  overview by Konstantin Vorontsov (in russian) - https://youtube.com/watch?v=kfhqzkcfMqI
  overview by Victor Kantor (in russian) - https://youtube.com/watch?v=Te_6TqEhyTI
  overview by Vladimir Gulin (in russian) - https://youtube.com/watch?v=5ir_fCgzfLM
  overview by Alexey Dral (in russian) - https://youtube.com/watch?v=MLljnzsz9Dk


  overview of Yandex Recommender System by Michael Rozner - https://youtube.com/watch?v=VgioTyMyJus
  overview of Yandex.Disco by Michael Rozner (in russian) - https://youtube.com/watch?v=JKTneRi2vn8
  overview of Yandex.Zen by Igor Lifar and Dmitry Ushanov (in russian) - https://youtube.com/watch?v=iGAMPnv-0VY


  "Deep Learning for Recommender Systems" by Alexandros Karatzoglou - https://youtube.com/watch?v=KZ7bcfYGuxw
  "Deep Learning for Recommender Systems" tutorial by Alexandros Karatzoglou and Balazs Hidasi - https://slideshare.net/kerveros99/deep-learning-for-recommender-systems-recsys2017-tutorial

  "Deep Learning for Personalized Search and Recommender Systems" from LinkedIn and Airbnb - https://youtube.com/watch?v=0DYQzZp68ok


  ACM RecSys conference - https://youtube.com/channel/UC2nEn-yNA1BtdDNWziphPGA



---
### challenges

  - diversity vs accuracy
  - popularity vs personalization
  - item fatigue vs freshness (repeating items)
  - serendepity (how much to surprise user)
  - predicting future vs influencing user


  [http://quora.com/Recommendation-Systems/What-developments-have-occurred-in-recommender-systems-after-the-Netflix-Prize]:
  - Implicit feedback from usage has proven to be a better and more reliable way to capture user preference.
  - Rating prediction is not the best formalization of the "recommender problem". Other approaches, and in particular personalized Learning to Rank, are much more aligned with the idea of recommending the best item for a user.  
  - It is important to find ways to balance the trade-off between exploration and exploitation. Approaches such as Multi-Armed Bandit algorithms offer an informed way to address this issue.
  - Issues such as diversity and novelty can be as important as relevance.
  - It is important to address the presentation bias caused by users only being able to give feedback to those items previously decided where good for them.
  - The recommendation problem is not only a two dimensional problem of users and items but rather a multi-dimensional problem that includes many contextual dimensions such as time of the day or day of the week. Algorithms such as Tensor Factorization or Factorization Machines come in very handy for this.
  - Users decide to select items not only based on how good they think they are, but also based on the possible impact on their social network. Therefore, social connections can be a good source of data to add to the recommendation system.
  - It is not good enough to design algorithms that select the best items for users, these items need to be presented with the right form of explanations for users to be attracted to them.



---
### interesting papers

[selected papers](https://yadi.sk/d/RtAsSjLG3PhrT2)

----
#### ["DropoutNet: Addressing Cold Start in Recommender Systems"](https://papers.nips.cc/paper/7081-dropoutnet-addressing-cold-start-in-recommender-systems) Volkovs, Yu, Poutanen
>	"Latent models have become the default choice for recommender systems due to their performance and scalability. However, research in this area has primarily focused on modeling user-item interactions, and few latent models have been developed for cold start. Deep learning has recently achieved remarkable success showing excellent results for diverse input types. Inspired by these results we propose a neural network based latent model called DropoutNet to address the cold start problem in recommender systems. Unlike existing approaches that incorporate additional content-based objective terms, we instead focus on the optimization and show that neural network models can be explicitly trained for cold start through dropout."

>	"Our approach is based on the observation that cold start is equivalent to the missing data problem where preference information is missing. Hence, instead of adding additional objective terms to model content, we modify the learning procedure to explicitly condition the model for the missing input. The key idea behind our approach is that by applying dropout to input mini-batches, we can train DNNs to generalize to missing input. By selecting an appropriate amount of dropout we show that it is possible to learn a DNN-based latent model that performs comparably to state-of-the-art on warm start while significantly outperforming it on cold start. The resulting model is simpler than most hybrid approaches and uses a single objective function, jointly optimizing all components to maximize recommendation accuracy."

>	"Training with dropout has a two-fold effect: pairs with dropout encourage the model to only use content information, while pairs without dropout encourage it to ignore content and simply reproduce preference input. The net effect is balanced between these two extremes. The model learns to reproduce the accuracy of the input latent model when preference data is available while also generalizing to cold start."

>	"An additional advantage of our approach is that it can be applied on top of any existing latent model to provide/enhance its cold start capability. This requires virtually no modification to the original model thus minimizing the implementation barrier for any production environment that’s already running latent models."

	-- https://youtu.be/YSQqHlQwQDY?t=1h44m18s (Ushanov) `in russian`
	-- https://github.com/HongleiXie/DropoutNet


Liang, Krishnan, Hoffman, Jebara - "Variational Autoencoders for Collaborative Filtering" [https://arxiv.org/abs/1802.05814]
  `Netflix`
>	"We extend variational autoencoders to collaborative filtering for implicit feedback. This non-linear probabilistic model enables us to go beyond the limited modeling capacity of linear factor models which still largely dominate collaborative filtering research. We introduce a generative model with multinomial likelihood and use Bayesian inference for parameter estimation. Despite widespread use in language modeling and economics, the multinomial likelihood receives less attention in the recommender systems literature. We introduce a different regularization parameter for the learning objective, which proves to be crucial for achieving competitive performance. Remarkably, there is an efficient way to tune the parameter using annealing. The resulting model and learning algorithm has information-theoretic connections to maximum entropy discrimination and the information bottleneck principle. Empirically, we show that the proposed approach significantly outperforms several state-of-the-art baselines, including two recently-proposed neural network approaches, on several real-world datasets. We also provide extended experiments comparing the multinomial likelihood with other commonly used likelihood functions in the latent factor collaborative filtering literature and show favorable results. Finally, we identify the pros and cons of employing a principled Bayesian inference approach and characterize settings where it provides the most significant improvements."

>	"Recommender systems is more of a "small data" than a "big data" problem."  
>	"VAE generalizes linear latent factor model and recovers Gaussian matrix factorization as a special linear case. No iterative procedure required to rank all the items given a user's watch history - only need to evaluate inference and generative functions."  
>	"We introduce a regularization parameter for the learning objective to trade-off the generative power for better predictive recommendation performance. For recommender systems, we don't necessarily need all the statistical property of a generative model. We trade off the ability of performing ancestral sampling for better fitting the data."  
	-- https://youtube.com/watch?v=gRvxr47Gj3k (Liang)
	-- https://github.com/dawenl/vae_cf>


Stern, Herbrich, Graepel - "Matchbox: Large Scale Bayesian Recommendations" [http://research.microsoft.com/apps/pubs/default.aspx?id=79460]
  `Microsoft`
	"We present a probabilistic model for generating personalised recommendations of items to users of a web service. The Matchbox system makes use of content information in the form of user and item meta data in combination with collaborative filtering information from previous user behavior in order to predict the value of an item for a user. Users and items are represented by feature vectors which are mapped into a low-dimensional ‘trait space’ in which similarity is measured in terms of inner products. The model can be trained from different types of feedback in order to learn user-item preferences. Here we present three alternatives: direct observation of an absolute rating each user gives to some items, observation of a binary preference (like/ don’t like) and observation of a set of ordinal ratings on a userspecific scale. Efficient inference is achieved by approximate message passing involving a combination of Expectation Propagation and Variational Message Passing. We also include a dynamics model which allows an item’s popularity, a user’s taste or a user’s personal rating scale to drift over time. By using Assumed-Density Filtering for training, the model requires only a single pass through the training data. This is an on-line learning algorithm capable of incrementally taking account of new data so the system can immediately reflect the latest user preferences. We evaluate the performance of the algorithm on the MovieLens and Netflix data sets consisting of approximately 1,000,000 and 100,000,000 ratings respectively. This demonstrates that training the model using the on-line ADF approach yields state-of-the-art performance with the option of improving performance further if computational resources are available by performing multiple EP passes over the training data."
	-- http://videolectures.net/ecmlpkdd2010_graepel_mlm/ (21:05) (Graepel)


Zaheer, Ahmed, Smola - "Latent LSTM Allocation: Joint Clustering and Non-Linear Dynamic Modeling of Sequential Data" [http://proceedings.mlr.press/v70/zaheer17a/zaheer17a.pdf]
  `Amazon`
	"Recurrent neural networks, such as LSTM networks, are powerful tools for modeling sequential data like user browsing history or natural language text. However, to generalize across different user types, LSTMs require a large number of parameters, notwithstanding the simplicity of the underlying dynamics, rendering it uninterpretable, which is highly undesirable in user modeling. The increase in complexity and parameters arises due to a large action space in which many of the actions have similar intent or topic. In this paper, we introduce Latent LSTM Allocation for user modeling combining hierarchical Bayesian models with LSTMs. In LLA, each user is modeled as a sequence of actions, and the model jointly groups actions into topics and learns the temporal dynamics over the topic sequence, instead of action space directly. This leads to a model that is highly interpretable, concise, and can capture intricate dynamics. We present an efficient Stochastic EM inference algorithm for our model that scales to millions of users/documents. Our experimental evaluations show that the proposed model compares favorably with several state-of-the-art baselines."
	-- https://vimeo.com/240608072 (Zaheer)
	-- https://youtube.com/watch?v=ofaPq5aRKZ0 (Smola)


Zheng, Zaheer, Ahmed, Wang, Xing, Smola - "State Space LSTM Models with Particle MCMC Inference" [https://arxiv.org/abs/1711.11179]
  `Amazon`
	"LSTM is one of the most powerful sequence models. Despite the strong performance, however, it lacks the nice interpretability as in state space models. In this paper, we present a way to combine the best of both worlds by introducing State Space LSTM models that generalizes the earlier work of combining topic models with LSTM. However we do not make any factorization assumptions in our inference algorithm. We present an efficient sampler based on sequential Monte Carlo method that draws from the joint posterior directly. Experimental results confirms the superiority and stability of this SMC inference algorithm on a variety of domains."


Hidasi, Karatzoglou, Baltrunas, Tikk - "Session-based Recommendations with Recurrent Neural Networks" [http://arxiv.org/abs/1511.06939]
	"We apply recurrent neural networks on a new domain, namely recommendation system. Real-life recommender systems often face the problem of having to base recommendations only on short session-based data (e.g. a small sportsware website) instead of long user histories (as in the case of Netflix). In this situation the frequently praised matrix factorization approaches are not accurate. This problem is usually overcome in practice by resorting to item-to-item recommendations, i.e. recommending similar items. We argue that by modeling the whole session, more accurate recommendations can be provided. We therefore propose an RNN-based approach for session-based recommendations. Our approach also considers practical aspects of the task and introduces several modifications to classic RNNs such as a ranking loss function that make it more viable for this specific problem. Experimental results on two data-sets show marked improvements over widely used approaches."
	"In this paper we applied a kind of modern recurrent neural network to new application domain: recommender systems. We chose task of session based recommendations, because it is a practically important area, but not well researched. We modified the basic GRU in order to fit the task better by introducing session-parallel mini-batches, mini-batch based output sampling and ranking loss function. We showed that our method can significantly outperform popular baselines that used for this task. We think that our work can be the basis of both deep learning applications in recommender systems and session based recommendations in general. We plan to train the network on automatically extracted item representation that is built on content of the item itself (e.g. thumbnail, video, text) instead of the current input."
	-- https://youtube.com/watch?v=M7FqgXySKYk (Karatzoglou)
	-- https://youtube.com/watch?v=Mw2AV12WH4s (Hidasi)
	-- http://blog.deepsystems.io/session-based-recommendations-rnn (in russian)
	-- https://github.com/yhs-968/pyGRU4REC


Cheng et al. - "Wide & Deep Learning for Recommender Systems" [http://arxiv.org/abs/1606.07792]
  `Google`
	"Generalized linear models with nonlinear feature transformations are widely used for large-scale regression and classification problems with sparse inputs. Memorization of feature interactions through a wide set of cross-product feature transformations are effective and interpretable, while generalization requires more feature engineering effort. With less feature engineering, deep neural networks can generalize better to unseen feature combinations through low-dimensional dense embeddings learned for the sparse features. However, deep neural networks with embeddings can over-generalize and recommend less relevant items when the user-item interactions are sparse and high-rank. In this paper, we present Wide & Deep learning—jointly trained wide linear models and deep neural networks—to combine the benefits of memorization and generalization for recommender systems. We productionized and evaluated the system on Google Play, a commercial mobile app store with over one billion active users and over one million apps. Online experiment results show that Wide & Deep significantly increased app acquisitions compared with wide-only and deep-only models. We have also open-sourced our implementation in TensorFlow."
	-- https://youtube.com/watch?v=NV1tkZ9Lq48 (Cheng)
	-- https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html
	-- https://www.tensorflow.org/tutorials/wide_and_deep
	-- https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/wide_n_deep_tutorial.py


Covington, Adams, Sargin - "Deep Neural Networks for YouTube Recommendations" [http://research.google.com/pubs/pub45530.html]
  `Google`
	"YouTube represents one of the largest scale and most sophisticated industrial recommendation systems in existence. In this paper, we describe the system at a high level and focus on the dramatic performance improvements brought by deep learning. The paper is split according to the classic two-stage information retrieval dichotomy: first, we detail a deep candidate generation model and then describe a separate deep ranking model. We also provide practical lessons and insights derived from designing, iterating and maintaining a massive recommendation system with enormous userfacing impact."
	"We have described our deep neural network architecture for recommending YouTube videos, split into two distinct problems: candidate generation and ranking. Our deep collaborative filtering model is able to effectively assimilate many signals and model their interaction with layers of depth, outperforming previous matrix factorization approaches used at YouTube. There is more art than science in selecting the surrogate problem for recommendations and we found classifying a future watch to perform well on live metrics by capturing asymmetric co-watch behavior and preventing leakage of future information. Withholding discrimative signals from the classifier was also essential to achieving good results - otherwise the model would overfit the surrogate problem and not transfer well to the homepage. We demonstrated that using the age of the training example as an input feature removes an inherent bias towards the past and allows the model to represent the time-dependent behavior of popular of videos. This improved offline holdout precision results and increased the watch time dramatically on recently uploaded videos in A/B testing. Ranking is a more classical machine learning problem yet our deep learning approach outperformed previous linear and tree-based methods for watch time prediction. Recommendation systems in particular benefit from specialized features describing past user behavior with items. Deep neural networks require special representations of categorical and continuous features which we transform with embeddings and quantile normalization, respectively. Layers of depth were shown to effectively model non-linear interactions between hundreds of features. Logistic regression was modified by weighting training examples with watch time for positive examples and unity for negative examples, allowing us to learn odds that closely model expected watch time. This approach performed much better on watch-time weighted ranking evaluation metrics compared to predicting click-through rate directly."
	-- https://youtube.com/watch?v=WK_Nr4tUtl8 (Covington)


"Personalization for Google Now: User Understanding and Application to Information Recommendation and Exploration" [http://dl.acm.org/citation.cfm?id=2959192]
  `Google`
	"At the heart of any personalization application, such as Google Now, is a deep model for users. The understanding of users ranges from raw history to lower dimensional reductions like interest, locations, preferences, etc. We will discuss different representations of such user understanding. Going from understanding to application, we will talk about two broad applications recommendations of information and guided exploration - both in the context of Google Now. We will focus on such applications from an information retrieval perspective. Information recommendation then takes the form of biasing information retrieval, in response to a query or, in the limit, in a query-less application. Somewhere in between lies broad declaration of user intent, e.g., interest in food, and we will discuss how personalization and guided exploration play together to provide a valuable tool to the user. We will discuss valuable lessons learned along the way."
	-- https://youtube.com/watch?v=X9Fsn1j1CE8 (Thakur)


"Two Decades of Recommender Systems at Amazon.com" [https://www.computer.org/csdl/mags/ic/2017/03/mic2017030012.html]
  `Amazon`


"Recommending Items to More than a Billion People" [https://code.facebook.com/posts/861999383875667/recommending-items-to-more-than-a-billion-people/]
  `Facebook`
