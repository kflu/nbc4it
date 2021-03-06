/**
 * \file classifier.cpp
 * \author Kefei Lu
 * \brief Implementations of Classifier and related classes.
 */

#include "classifier.h"

#define PI 3.1415926
#define __CLASSIFICATION_DEBUG__
//#define __CLASSIFICATION_DEBUG_VERBOSE__

Classifier::Classifier( const Dataset& dataset,
	    const size_t classIndex,
	    const bool useAllAtt)
	    //const RSeed seed,
	    //const double tt_ratio)
{
    _bindedDataset = &dataset;
    _classIndex = classIndex;
    _useAllAtt = useAllAtt;
    //_seed = seed;
    _onlyTheseAtt.clear();
    perf_clear();

    // empty_tt_set();
    init_tt_set();

    //ran_tt_set();
}

void 
Classifier::init_tt_set(void)
{
    train_set().clear();
    test_set().clear();

    size_t nInst = dataset().num_of_inst();
    for( size_t i=0;i<nInst;i++ ) {
	train_set().push_back(i);
    }
    test_set() = train_set();
    return;
}

void 
Classifier::test(void)
{
    assert(!test_set().empty());
    assert(!train_set().empty());

    conf().clear();
    size_t nTest = test_set().size();
    size_t nClass = dataset().get_att_desc( class_index() ).possible_value_vector().size();

    // Init. the Confusion Mat.
    {
	vector<size_t> tmp(nClass,.0);
	for( size_t i=0;i<nClass;i++ ) {
	    conf().push_back(tmp);
	}
	// Now it should be nClass x nClass matrix with zeros
    }

    // Init. the trust vector
    trust().clear();
    trust().resize(nClass, 0);

    // Begin testing
    for( size_t i=0;i<nTest;i++ ) {
#ifdef __CLASSIFICATION_DEBUG__
#ifdef   __CLASSIFICATION_DEBUG_VERBOSE__
	fprintf(stdout, "(D) Testing %d-th instance (total %d).\n",
		i+1,nTest);
#endif
#endif
	const Instance& inst = dataset()[test_set()[i]];
	const size_t ci = class_index();

	if (inst[ci].unknown) continue;

	conf()[classify_inst(inst)][inst[ci].value.nom] ++ ;
    }

    // Calculate accuracy:
    {
	size_t sum=0;
	for (size_t i=0;i<nClass;i++) {
	    // The diagonal is the correct clssified inst
	    sum += conf()[i][i];
	}
	accuracy() = (double)sum / nTest;
#ifdef __CLASSIFICATION_DEBUG__
	fprintf(stdout, "(D) Accuracy: %g.\n",accuracy());
	fprintf(stdout, "(D) Correct: %d, Total: %d\n",sum,nTest);
#endif
    }
    
    // Calculate trust:

    // for each row:
    for( size_t r=0; r<nClass; r++ ) {
        // get row sum
        size_t sum = 0;
        for( size_t c=0; c<nClass; c++ ) {
            sum += conf()[r][c];
        }
	if (conf()[r][r]==0 && sum==0) {
	    trust()[r] = 0;
	} else {
	    trust()[r] = conf()[r][r] / (double)sum;
	}
    }
#ifdef __CLASSIFICATION_DEBUG__
    show_conf();
    show_trust();
#endif
}

void 
show_conf(const Classifier& c, const ConfMatr& conf)
{
    fprintf(stdout, "(I) Confusion Matrix:\n");
    size_t nRow = conf.size();
    size_t nCol = conf[0].size();
    for ( size_t i=0;i<nRow;i++ ) {
	fprintf(stdout, "(I) ... ");
	for ( size_t j=0;j<nCol;j++ ) {
	    fprintf(stdout, "%7d", conf[i][j]);
	}
	fprintf(stdout, " ... %s \n", c.get_class_desc().map(i).c_str());
    }
}

/* this is for the average confusion matrix */
void 
show_conf(const Classifier& c, const vector< vector<double> >& conf)
{
    fprintf(stdout, "(I) Confusion Matrix:\n");
    size_t nRow = conf.size();
    size_t nCol = conf[0].size();
    for ( size_t i=0;i<nRow;i++ ) {
	fprintf(stdout, "(I) ... ");
	for ( size_t j=0;j<nCol;j++ ) {
	    fprintf(stdout, "%7.3f  ", conf[i][j]);
	}
	fprintf(stdout, " ... %s \n", c.get_class_desc().map(i).c_str());
    }
}

void 
show_trust(const Classifier& c,const vector<double>& trust)
{
    fprintf(stdout, "(I) The trust on class label:\n");
    size_t n = trust.size();
    for (size_t i=0;i<n;i++) {
	fprintf(stdout, "(I) ... %.6f: %s(%d)\n", 
		trust[i], c.get_class_desc().map(i).c_str(), i);
    }
}

NominalType 
StatisticsClassifier::
classify_inst(const Instance& inst, double* maxProb) const 
{
    const size_t nClass = 
	dataset().get_att_desc( class_index() ).possible_value_vector().size();
    size_t curMaxClassIndex = 0;
    double curMaxProb = -1;
    double tmp=0;
    if (maxProb) {
	for( size_t i=0;i<nClass;i++ ) {
	    tmp = a_posteriori(i, inst);
	    if ( tmp > curMaxProb ) {
		curMaxProb = tmp;
		curMaxClassIndex = i;
	    }
	}
	*maxProb = curMaxProb;
    }
    else {
	for( size_t i=0;i<nClass;i++ ) {
	    tmp = likelihood(i, inst);
	    if ( tmp > curMaxProb ) {
		curMaxProb = tmp;
		curMaxClassIndex = i;
	    }
	}
    }
    return curMaxClassIndex;
}

inline
double
StatisticsClassifier::
likelihood(const NominalType c, const Instance& inst) const 
{
    return prob_inst_on_class(inst,c) * pClass()[c];
}

double 
StatisticsClassifier::
a_posteriori(const NominalType c, const Instance& inst) const 
{
    const size_t nClass = get_class_desc().possible_value_vector().size();
    double pInst = 0;
    for (size_t i=0;i<nClass;i++) {
	pInst += prob_inst_on_class(inst,i) * pClass()[i];
    }
    return likelihood(c,inst) / pInst;
}

double
StatisticsClassifier::
est_class_prob(const size_t c_index) const
{
    assert(!train_set().empty());
    assert(!test_set().empty());

    const AttDesc & classDesc = get_class_desc();
    size_t nClass = classDesc.possible_value_vector().size();
    assert(c_index<nClass);

    // count num of instance belongs to class i:
    const size_t nTrain = train_set().size();
    size_t sum = 0;
    // for each inst in train set
    for ( size_t j=0;j<nTrain;j++ ) {
	const Attribute& c = dataset()[train_set()[j]][this->class_index()];
	if (c.unknown) {continue;}
	if (c.value.nom == c_index) {sum ++;}
    }
    /** Handling zero-instance issue (no inst. belongs to this class). */
    if (sum==0) {
	fprintf(stderr, "(W) No Training instance belongs to class %s (%d). "
		"Probability set to 0.\n",
		get_class_desc().map(c_index).c_str(), c_index);
	return 0.0;
    }
    return sum/(double)nTrain;
}

double 
NaiveBayesClassifier::
att_prob_on_class(const ValueType& value, const size_t att_i, const size_t class_j) const
{
    // Check if training/testing set has been specified before.
    assert(!train_set().empty());
    assert(!test_set().empty());

    // Check if train() has been called before.
    assert(!pClass().empty());

    return _attDistrOnClass.prob(value, att_i, class_j);
}

void 
StatisticsClassifier::
train(void)
{
    fprintf(stdout, "(I) StatisticsClassifier: Training the model...\n");
    assert(!train_set().empty());
    assert(!test_set().empty());

    // Obtaining _pClass:
    pClass().clear();
    size_t nClass = get_class_desc().possible_value_vector().size();
    for ( size_t i=0;i<nClass;i++ ) {
	pClass().push_back( est_class_prob(i) );
    }
#ifdef __CLASSIFICATION_DEBUG__
    fprintf(stdout, "(I) Priori probability of class:\n");
    for (size_t i=0;i<nClass;i++) {
	fprintf(stdout, "(I) ... %.7f (%s)\n",
		pClass().at(i), get_class_desc().map(i).c_str() );
    }
#endif
}

void
NaiveBayesClassifier::
train(void)
{
    StatisticsClassifier::train();
    fprintf(stdout, "(I) NaiveBayesClassifier: Training the model...\n");

    assert(!train_set().empty());
    assert(!test_set().empty());

    // Obtaining _attDistrOnClass:
    const size_t nAtt = dataset().num_of_att();
    const size_t ci = class_index();
    size_t nClass = get_class_desc().possible_value_vector().size();
    for ( size_t i=0; i<nAtt; i++ ) {
#ifdef __CLASSIFICATION_DEBUG__
#ifdef   __CLASSIFICATION_DEBUG_VERBOSE__
	fprintf(stdout, "(D) NaiveBayesClassifier: training on %d-th attribute (total: %d).\n", i+1, nAtt);
#endif
#endif
	if ( i == ci ) continue;
	for ( size_t j=0; j<nClass; j++ ) {
	    calc_distr_for_att_on_class(i,j);
	}
    }
}

//Distribution* 
void
NaiveBayesClassifier::
calc_distr_for_att_on_class(size_t att_i, size_t class_j)
{
    const Dataset& ds = dataset();
    const size_t nInst = ds.num_of_inst();
    const size_t ci = class_index();
    //const size_t nClass = ds.get_att_desc(ci).possible_value_vector().size();

    // att_i is not the class attribute:
    if (att_i == ci) {
	fprintf(stderr, "(E) Attribute must not be the class attribute.\n");
	exit(1);
    }

    // This value to later store the new'd distribution 
    // and will be returned.
    //Distribution* pDistr = NULL;
    Distribution*& pDistr = attDistrOnClass().table()[class_j][att_i];
    // find type of this att:
    const AttDesc& desc = ds.get_att_desc(att_i);
    if (desc.get_type() == ATT_TYPE_NUMERIC) {
	//pDistr = new NormalDistribution;
	double sum=0;
	double sq_sum=0;
	size_t nInstBelongsToThisClass=0;
	for (size_t i=0;i<nInst;i++) {
	    const Attribute& klass = ds[i][ci];
	    const Attribute& att = ds[i][att_i];
	    if (klass.unknown) continue;
	    if (att.unknown) continue;
	    if (klass.value.nom != class_j) continue;
	    sum += att.value.num;
	    sq_sum += pow(att.value.num,2);
	    nInstBelongsToThisClass ++;
	}
	if (nInstBelongsToThisClass==0) {
	    /** When no instances belongs to this class, the _pClass should have 
	     * been already set to 0. Set the corresponding conditional probability 
	     * to invalid to indicate that when evaluating this conditional 
	     * probability, 0 should be returned. */
	    ((NormalDistribution*)pDistr)->invalid() = 1;
	    return;
	}
	double meantmp = sum/nInstBelongsToThisClass;
	((NormalDistribution*)pDistr)->mean() = meantmp;
	((NormalDistribution*)pDistr)->var() = 
	    1.0 / (nInstBelongsToThisClass-1) *
	    (
	      sq_sum 
	      + nInstBelongsToThisClass * pow(meantmp,2.0)
	      - 2 * meantmp * sum
	    );
	return;
    }
    else if (desc.get_type() == ATT_TYPE_NOMINAL) {
	//pDistr = new NominalDistribution;
	size_t nPos = ds.get_att_desc(att_i).possible_value_vector().size();
	((NominalDistribution*)pDistr)->pmf().resize(nPos,0.0);
	size_t sum = 0; // total num of inst belongs to class_j
	for (size_t i=0;i<nInst;i++) {
	    const Attribute& klass = ds[i][ci];
	    const Attribute& att = ds[i][att_i];
	    if (klass.unknown) continue;
	    if (att.unknown) continue;
	    if (klass.value.nom != class_j) continue;
	    sum ++;
	    ((NominalDistribution*)pDistr)->pmf()[att.value.nom] ++;
	}
	// Handle zero sum issue and so on.
	bool zero_issue=0;
	if (sum==0) {
	    /** When no instances belongs to this class, for Nominal type attributes, 
	     * we can assume the possible values of this attribute is equally likely 
	     * to be chosen. So this zero-instance issue can be addressed the same 
	     * way as the zero possibility issue stated as later. So simply set zero_issue 
	     * flags to 1. */
	    zero_issue = 1;
	}
	for (size_t i=0;i<nPos;i++) {
	    if ( ((NominalDistribution*)pDistr)->pmf()[i] != 0 )
		continue;
	    zero_issue = 1;
	}
	/** Handle the zero possibility issue.
	 *
	 * p{A_j|C_i} = N(A_j,C_i) / N(C_i)
	 *
	 * if N(A_j,C_i) == 0, means there's no such instance
	 * having A_j value and belongs to class C_i.
	 * This can be handled as:
	 *
	 * \verbatim
	                N(A_j,C_i) + 1
	 p{A_j|C_i} = ----------------- ,
	                N(C_i) + nPos
	   \endverbatim
	 *
	 * where nPos is the num of possible values of this 
	 * attribute.
	 *
	 * For example, 0/3, 3/3 will become 1/5, 4/5; 
	 * 0/3, 1/3, 2/3 will become 1/6, 2/6, 3/6. */
	for (size_t i=0;i<nPos;i++) {
	    if (!zero_issue) {
		((NominalDistribution*)pDistr)->pmf()[i] /= (double)sum;
	    } else {
		((NominalDistribution*)pDistr)->pmf()[i] = (double)(((NominalDistribution*)pDistr)->pmf()[i] + 1) / (sum + nPos);
	    }
	}
	return;
    }
    fprintf(stderr, "(E) Unsupported attribute type: %s(%d).\n",
	    desc.map(desc.get_type()).c_str(),desc.get_type());
    exit(1);
}

void 
NaiveBayesClassifier::
bind_dataset(const Dataset& dataset)
{
    Classifier::bind_dataset(dataset);
    attDistrOnClass().init_table();
}

const double 
NaiveBayesClassifier::
prob_inst_on_class( const Instance& inst, const NominalType c ) const
{
    const size_t nAtt = dataset().num_of_att();
    const size_t ci = class_index();
    double product = 1;
    if ( useAllAtt() ) {
	for (size_t i=0;i<nAtt;i++) {
	    if (ci == i) continue;
	    if (inst[i].unknown) continue;
	    product *= att_prob_on_class(inst[i].value, i, c);
	}
    }
    else {
	for (size_t i=0;i<only_these_att().size();i++) {
	    size_t ii = only_these_att().at(i);
	    if (ci == ii) continue;
	    if (inst[ii].unknown) continue;
	    product *= att_prob_on_class(inst[ii].value, ii, c);
	}
    }
    return product;
}

void
AttDistrOnClass::
init_table(void)
{
    if (!_classifier) {
	fprintf(stderr, "(E) No binding classifier.\n");
	exit(1);
    }

    const Classifier & c = classifier();
    size_t nAtt = c.dataset().num_of_att() - 1;
    size_t cIndex = c.class_index();
    size_t nClass = c.get_class_desc().possible_value_vector().size();

    {
	Distribution* tmp;
	table().clear();
	table().resize(nClass);
	for (size_t j=0;j<nClass;j++) {
	    table()[j].clear();
	    table()[j].resize(nAtt);
	    for (size_t i=0;i<nAtt+1;i++) {
		if (i==cIndex) continue;
		const AttDesc& desc = c.dataset().get_att_desc(i);
		if (desc.get_type() == ATT_TYPE_NUMERIC) {
		    tmp = new NormalDistribution;
		}
		else if (desc.get_type() == ATT_TYPE_NOMINAL) {
		    tmp = new NominalDistribution;
		}
		else {
		    fprintf(stderr, "(E) Unsupported type: %s (%d).\n",
			    desc.map(desc.get_type()).c_str(), desc.get_type());
		}
		table()[j][i] = tmp;
	    }
	}
    }
}

AttDistrOnClass::
~AttDistrOnClass()
{
    // Free all pointers
    if (_table.empty()) return;

    size_t nRow = _table.size();
    for (size_t i=0;i<nRow;i++) {
	for (size_t j=0;j<_table[i].size();j++) {
	    if (_table[i][j]) {
		delete _table[i][j];
		_table[i][j] = NULL;
	    }
	}
    }

    _table.clear();
    return;
}

bool float_eq(const double v1, const double v2)
{
    if (fabs(v1-v2) <= DBL_MIN) return 1;
    return 0;
}

const double 
NormalDistribution::
prob(const ValueType value) const
{
    if (invalid()) return .0;
    if ( float_eq(var(),0) ) {
	if (float_eq(value.num,mean())) return 1-DBL_MIN;
	return DBL_MIN;
    }
    return (1.0/sqrt(2*PI*var())) * exp( - pow(value.num-mean(),2.0) / (2.0*var()) );
}
