/**
 * \file classifier.cpp
 * \author Kefei Lu
 * \brief Implementations of Classifier and related classes.
 */

#include "classifier.h"

#define __CLASSIFICATION_DEBUG__

Classifier::Classifier( const Dataset* dataset,
	    const size_t classIndex,
	    const bool useAllAtt)
	    //const RSeed seed,
	    //const double tt_ratio)
{
    _bindedDataset = dataset;
    _classIndex = classIndex;
    _useAllAtt = useAllAtt;
    //_seed = seed;
    _onlyTheseAtt.clear();
    perf_clear();

    empty_tt_set();

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

#ifdef ______XXXXXXXXXXXXX__________
/**
 * \deprecated 
 * This member is deprecated. It belongs to cross
 * validation, which should be a separated class
 */
void 
Classifier::ran_tt_set(void)
{
    fprintf(stdout, "(I) Randomizing T/T dataset...\n");

    train_set().clear();
    test_set().clear();

    size_t nInst = dataset().num_of_inst();

    size_t nTrain = nInst * tt_ratio();
    size_t nTest = nInst - nTrain;

    vector<short> used(nInst,0);
    size_t tmp=0;
    /* gen. training set */
    for (size_t i=0;i<nTrain;i++ ) {
	while (1) {
	    tmp = rand() % nTrain;
	    if (!used[tmp]) {
		train_set().push_back(tmp);
		used[tmp]++;
		break;
	    }
	}
    }
    // Sort into increasing order.
    sort( train_set().begin(), train_set().end() );

    /* the rest is testing set */
    for (size_t i=0;i<nInst;i++) {
	if (!used[i]) {
	    test_set().push_back(i);
	    used[i]++;
	}
    }

#ifdef __CLASSIFICATION_DEBUG__
    /* verify that every inst is used in either training
     * or testing set and is only used once. */
    assert(nTest==test_set().size());
    assert(nTrain==train_set().size());
    for (size_t i=0;i<nInst;i++) {
	assert(used[i]==1);
    }
#endif
}
#endif

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
	vector<double> tmp(nClass,.0);
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
	const Instance& inst = dataset()[test_set()[i]];
	Attribute klass = inst[class_index()];

	assert( !klass.unknown );

	NominalType est_c = classify_inst( inst );
	NominalType true_c = inst[class_index()].value.nom;
	conf()[est_c][true_c]++;
    }
    // Normalize the conf matrix
    // for each row:
    for( size_t r=0; r<nClass; r++ ) {
	// get row sum
	size_t sum = 0;
	for( size_t c=0; c<nClass; c++ ) {
	    sum += conf()[r][c];
	}
	// Normalize each column.
	for( size_t c=0; c<nClass; c++ ) {
	    conf()[r][c] /= sum;
	    if (c==r) {
		// this element is trust.
		trust()[c] = conf()[r][c];
	    }
	}
    }

    // print_performance();
}


NominalType 
StatisticsClassifier::classify_inst(const Instance& inst, double* maxProb)
{
    size_t nClass = 
	dataset().get_att_desc( class_index() ).possible_value_vector().size();
    size_t curMaxClassIndex = 0;
    double curMaxProb = -1;
    for( size_t i=0;i<nClass;i++ ) {
	double tmp = a_posteriori(i, inst);
	if ( tmp > curMaxProb ) {
	    curMaxProb = tmp;
	    curMaxClassIndex = i;
	}
    }

    if (maxProb)
	*maxProb = curMaxProb;

    return curMaxClassIndex;
}

double
NaiveBayesClassifier::
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
	fprintf(stderr, "(W) No Training instance belongs to class %s (%d).\n",
		get_class_desc().map(c_index).c_str(), c_index);
    }
    return sum/nTrain;
}

double 
NaiveBayesClassifier::
est_att_prob_on_class(const ValueType& value, const size_t att_i, const size_t class_j) const
{
    // Check if training/testing set has been specified before.
    assert(!train_set().empty());
    assert(!test_set().empty());

    // Check if train() has been called before.
    assert(!pClass().empty());

    return _attDistrOnClass.prob(value, att_i, class_j);
}

void
NaiveBayesClassifier::
train(void)
{
    fprintf(stdout, "(I) Training the model...\n");

    assert(!train_set().empty());
    assert(!test_set().empty());

    pClass().clear();
    //distrAttOnClass().clear();

    // Obtaining _pClass:
    size_t nClass = get_class_desc().possible_value_vector().size();
    for ( size_t i=0;i<nClass;i++ ) {
	pClass().push_back( est_class_prob(i) );
    }

    // Obtaining _attDistrOnClass:

}

double 
NaiveBayesClassifier::
a_posteriori(const NominalType c, const Instance& inst)
{
    return 0;
}

void 
NaiveBayesClassifier::
bind_dataset(const Dataset& dataset)
{
    Classifier::bind_dataset(dataset);
    attDistrOnClass().init_table();
}

void
AttDistrOnClass::
init_table(void)
{
    assert(_classifier);

    const Classifier & c = classifier();
    size_t nAtt = c.dataset().num_of_att() - 1;
    size_t cIndex = c.class_index();
    size_t nClass = c.get_class_desc().possible_value_vector().size();

    {
	vector<Distribution*> distr;
	Distribution* tmp;
	for (size_t i=0;i<nAtt+1;i++) {
	    if (i==cIndex) continue;
	    const AttDesc& desc = c.dataset().get_att_desc(i);
	    if (desc.get_type() == ATT_TYPE_NUMERIC) {
		tmp = new NormalDistribution;
		distr.push_back(tmp);
	    }
	    else if (desc.get_type() == ATT_TYPE_NOMINAL) {
		tmp = new NominalDistribution;
		distr.push_back(tmp);
	    }
	    else {
		fprintf(stderr, "(E) Unsupported type: %s (%d).\n",
			desc.map(desc.get_type()).c_str(), desc.get_type());
	    }
	}
	for (size_t i=0;i<nClass;i++) {
	    table().push_back(distr);
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
