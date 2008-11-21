/**
 * \file classifier.h
 * \brief Classifier and related classes header file.
 * \author Kefei Lu
 */

#ifndef __CLASSIFIER_H__
#define __CLASSIFIER_H__

#include "common.h"
#include "dataset.h"

/**
 * Confusion Matrix.
 *
 * \verbatim
                          true_class:
			    ----->
                  _                     _ 
                 |     1   2  3  ...  N  |  
                 |  1                    | |
                 |  2                    | | est_class:
   ConfMatrix =  |  3       {p_i_j}      | |
                 |  ...                  | V
                 |_ N                   _|  
  
   where p_i_j = Pr{true_c = j | est_c = i}
               = Pr{true_c = j;est_c = i} / Pr{est_c = i}
               = Num{true_c = j;est_c = i} / Num{est_c = i} 
   \endverbatim
 */
typedef vector< vector<double> > ConfMatr;

/**
 * Random seed type.
 *
 * According to GNU libc, random seed is of type uint.
 */
typedef unsigned int	RSeed;


/**
 * Basic classifier class.
 *
 * A simple classifier that supports cross validation on a single 
 * dataset.
 */
class Classifier {
    private:
	/* ========== Classifier Settings ================ */
	/** Ratio of num of training and testing inst */
	//double 		_train_test_ratio;

	vector<size_t>	_test_set;
	vector<size_t>	_train_set;

	/** The dataset that this classifier is binded to 
	 *  This must be const */
	const Dataset* 	_bindedDataset;
	/** The column of attributes that represents the class. 
	 *   NOTE that this must be a nominal type attribute */
	size_t		_classIndex;

	/**
	 * Only use these attributes specified by the indecs.
	 */
	vector<size_t>	_onlyTheseAtt;
	/** If use all the att on classification, default yes */
	bool		_useAllAtt;

    private:
	/**
	 * Classify the instance.
	 *
	 * Make classification on inst, and returns the type value.
	 * Note that inst may not be compatible with _bindedDataset, 
	 * and this will not be checked by the program.
	 *
	 * So this function is private and should only be used by 
	 * other methods.
	 */
	virtual NominalType 
	    classify_inst(const Instance& inst, double* maxProb=NULL) = 0;

    private:
	/* =============== Performances ================== */
	/** Accuracy of the whole classifier */
	double 		_accuracy;
	/** confusion matrix */
	ConfMatr	_conf;
	/** Trust for each class */
	vector<double>	_trust;

    public:
	//RSeed& seed(void) {return _seed;}
	//void srand(void) {::srand(seed());}
	//double& tt_ratio(void) {return _train_test_ratio;}
	vector<size_t>& train_set(void) {return _train_set;}
	const vector<size_t>& train_set(void) const {return _train_set;}
	vector<size_t>& test_set(void) {return _test_set;}
	const vector<size_t>& test_set(void) const {return _test_set;}

	const Dataset& dataset(void) const {assert(_bindedDataset);return *_bindedDataset;}
	/**
	 * Bind the dataset to this classifier.
	 *
	 * It's virtual because in the inherited Classifiers this method may be 
	 * redefined so that the binding action will trigger other action, like
	 * initializing the statistics matrix, etc.
	 *
	 * \sa NaiveBayesClassifier
	 */
	virtual void bind_dataset(const Dataset& dataset) {_bindedDataset=&dataset;}

	size_t & class_index(void) {return _classIndex;}
	const size_t & class_index(void) const {return _classIndex;}

	/** \sa _onlyTheseAtt */
	vector<size_t> & only_these_att(void) {return _onlyTheseAtt;};

	double& accuracy(void) {return _accuracy;}
	vector<double>& trust(void) {return _trust;}
	ConfMatr& conf(void) {return _conf;}

	/** Clear the performance parameters. */
	void perf_clear(void) {_accuracy=0;_trust.clear();_conf.clear();}

	/**
	 * Randomize the training and testing set.
	 */
	void ran_tt_set(void);

	/** 
	 * Initialize training / testing set to the whole dateset.
	 */
	void init_tt_set(void);

	void empty_tt_set() {train_set().clear();test_set().clear();}
	/**
	 * Get an AttDesc reference on the class attribute.
	 */
	const AttDesc& get_class_desc(void) const
	{
	    return dataset().get_att_desc(class_index());
	}

	/** 
	 * Train on training instances of _bindedDataset 
	 *
	 * Depends on detailed implementation.
	 */
	virtual void train(void) = 0;

	/** 
	 * Test on testing instances of _bindedDataset.
	 */
	void test(void);

	/** Print the performance statistics. 
	 * 
	 * Not implemented yet.*/
	void print_performance(void);
	

	Classifier( const Dataset* dataset = NULL,
	    const size_t classIndex = 0,
	    const bool useAllAtt = 1);
	    //const RSeed seed = 0,
	    //const double tt_ratio = 2.0 );
};

/** 
 * Classifier based on Maximum A Posteriori criteria.
 */
class StatisticsClassifier : public Classifier {
    public:
	NominalType classify_inst(const Instance& inst, double* maxProb=NULL);
	virtual double a_posteriori(const NominalType c, const Instance& inst) = 0;
};

/**
 * The distributioin's base class.
 *
 * There can be different kinds of distributions, each with a different
 * way to describe. Thus the base class only provide the interface an 
 * inherited class must implement: a method that returns a probability
 * when feed with a value.
 */
class Distribution {
    public:
	virtual const double prob(ValueType value) const = 0;
};

/**
 * Normal Distribution describer.
 *
 * Specified by the mean and variance.
 */
class NormalDistribution : public Distribution {
    private:
	NumericType _mean;
	NumericType _var;
    public:
	NumericType& mean() {return _mean;}
	const NumericType& mean() const {return _mean;}
	NumericType& var() {return _var;}
	const NumericType& var() const {return _var;}
	
	const double prob(const ValueType value) const;
};

/**
 * Nominal Distribution describer.
 */
class NominalDistribution : public Distribution {
    private:
	vector<double> _pmf;
    public:
	vector<double>& pmf() {return _pmf;}
	const vector<double>& pmf() const {return _pmf;}
	const double prob(const ValueType value) const;
};

/**
 * A table storing distribution of the attributes conditioned on class value.
 *
 * Element [r,c] corresponds to r-th class and c-th attribute, 
 * that is, the distribution information of the random variable of 
 * r-th attribute given class is c.
 *
 * \sa NaiveBayesClassifier::_attDistrOnClass
 */
class AttDistrOnClass {
    private:
	/**
	 * The binded Classifier.
	 */
	const Classifier* _classifier;
	/** 
	 * Distribution of attr conditioned on class.
	 *
	 * Element [r,c] corresponds to r-th class and c-th attribute, 
	 * that is, the distribution information of the random variable of 
	 * r-th attribute given class is c.
	 */
	vector< vector<Distribution*> > _table;
    public:
	~AttDistrOnClass();
	/*
	 * Initialize the attribute distribution on class table.
	 */
	void init_table();

	vector< vector<Distribution*> > & table() {return _table;}
	void bind_classifier(const Classifier& c) {_classifier = &c;}
	const Classifier& classifier(void) {assert(_classifier);return *_classifier;}
	const double prob(const ValueType& value, const size_t att_i, const size_t class_j) const
	{
	    assert(_classifier);
	    assert(!_table.empty());
	    assert(_table[class_j][att_i]);
	    return _table[class_j][att_i]->prob(value);
	}
};
/**
 * Naive Bayesian method.
 */
class NaiveBayesClassifier : public StatisticsClassifier {
    private:
	vector<double>	_pClass; ///< prob of a class.
	/** 
	 * Distribution of attr conditioned on class.
	 *
	 * Element [r,c] corresponds to r-th class and c-th attribute, 
	 * that is, the distribution information of the random variable of 
	 * r-th attribute given class is c.
	 *
	 * \sa AttDistrOnClass
	 */
	AttDistrOnClass _attDistrOnClass;

	/*
	 * The estimation functions are only used by train()
	 * to train the model.
	 */
	/**
	 * Estimate the i-th class's probability.
	 *
	 * ONLY use training instances.
	 */
	double est_class_prob(const size_t i) const;

	/**
	 * Estimate the conditional prob of i-th att value given j-th class.
	 *
	 * This method is based on the trained model, i.e. _pClass, _attDistrOnClass, 
	 * which is trained by the train() method. Don't use it before the model is 
	 * actually trained.
	 */
	virtual double est_att_prob_on_class(const ValueType& value, const size_t att_i, const size_t class_j) const;

	/**
	 * Calculate a Distribution for RV att_i conditioned on class_j.
	 *
	 * \return A new'd pointer to a Distribution (or the inherited 
	 * classes like NormalDistribution).
	 *
	 * NOTE that it'll return a new'd pointer because the distribution 
	 * may be an interitance of class Distribution. So use a Distribution* 
	 * to carry the return value. This should be standardly done by store 
	 * this pointer in the AttDistrOnClass table.
	 */
	virtual Distribution* calc_distr_for_att_on_class(size_t att_i, size_t class_j) const;

    public:
	virtual void bind_dataset(const Dataset& dataset);
	vector<double>& pClass(void) {return _pClass;}
	const vector<double>& pClass(void) const {return _pClass;}
	//vector< vector<double> >& distrAttOnClass(void) {return _distrAttOnClass;}
	AttDistrOnClass& attDistrOnClass(void) {return _attDistrOnClass;}

	/**
	 * Calculate the a posteriori probability in Naive Bayesian method.
	 *
	 * Here attributes are assumed to be independent to each other, so
	 *  that the joint prob. can be evaluated as the product of marginal 
	 *  prob. Futhurmore, the attributes' marginal probs are assumed to 
	 *  be normally distributed.
	 */
	double a_posteriori(const NominalType c, const Instance& inst);

	/*
	 * Train the model.
	 *
	 * Using train_class_prob() and train_att_prob_on_class() and put the 
	 * corresponding values in _pClass and _attDistrOnClass.
	 *
	 * Handles the issue in which the probability may be zero.
	 */
	void train(void);

	NaiveBayesClassifier() {attDistrOnClass().bind_classifier(*this);}
};

/**
 * Only used for half-way testing.
 */
class NaiveBayesClassifierFake : public StatisticsClassifier {
    private:
    public:
	void train(void)
	{
	    // do nothing.
	}
	double a_posteriori(const NominalType c, const Instance& inst)
	{
	    if (c==0) return 1.0;
	    return 0.0;
	}
};

#endif
