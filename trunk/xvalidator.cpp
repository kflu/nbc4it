/**
 * \file xvalidator.cpp
 * \author Kefei Lu
 * \brief Implementation of cross validator.
 */
#include "common.h"
#include "xvalidator.h"

#define __XVALIDATOR_DEBUG__

Xvalidator::
Xvalidator(Classifier* c, const size_t f, RSeed s)
{
    _binded_classifier = c;
    seed() = s;
    set_fold(f);
    init_randomIndex();
}

void
Xvalidator::
init_randomIndex()
{
    randomIndecs().clear();
    randomIndecs().resize(fold());
    const size_t nInst = classifier().dataset().num_of_inst();
    const size_t len = nInst / fold();
    const size_t len_last = len + nInst % fold();
    for ( size_t i=0;i<fold();i++ ) {
	randomIndecs().at(i).clear();
	if (i == fold()-1) {
	    randomIndecs().at(i).resize(len_last,0);
	} else {
	    randomIndecs().at(i).resize(len,0);
	}
    }
}

void 
Xvalidator::randomize(void) 
{
    fprintf(stdout, "(I) Randomizing the instances...\n");
    const size_t nInst = classifier().dataset().num_of_inst();
    vector<size_t> unused(nInst);
    for (size_t i=0;i<nInst;i++) {
	unused.at(i) = i;
    }

    // Init. and randomize the ran vector.
    vector<size_t> ran(nInst,0);
    for (size_t i=0;i<nInst;i++) {
	ran.at(i) = i;
    }
    random_shuffle( ran.begin(), ran.end() );
    // // randomize the ran vector
    // size_t tmp = 0;
    // for (size_t i=0;i<nInst;i++) {
    //     tmp = rand() % (nInst-i);
    //     ran.at(i) = tmp;
    //     // remove the tmp-th element from unused:
    //     unused.erase(unused.begin()+tmp);
    // }

    { // Assign ran vectors to the randomIndecs
	size_t i = 0;
	size_t j = 0;
	size_t k = 0;
	for (i=0;i<fold();i++) { // i-th randomIndecs vector
	    j = 0; // keep the current position in i-th randomIndecs vector
	    while ( j < randomIndecs().at(i).size() ) {
		randomIndecs().at(i).at(j) = ran.at(k);
		k++; // keep the current position in ran vector.
		j++;
	    }
	}
    }
    // Sort every randomIndecs vector.
    for (size_t i=0;i<fold();i++) {
	sort(randomIndecs().at(i).begin(),randomIndecs().at(i).end());
    }
}

static
double double_sum(const double v1, const double v2)
{
    return v1+v2;
}

static
size_t size_t_sum(const size_t v1, const size_t v2)
{
    return v1+v2;
}

static
vector<double>& vector_sum(vector<double>& v1, const vector<size_t>& v2)
{
    transform( v1.begin(), v1.end(), v2.begin(), v1.begin(), size_t_sum );
    return v1;
}

void 
Xvalidator::
xvalidate()
{
    srand(seed());
    randomize();
    Classifier& c = classifier();
    const size_t nInst = c.dataset().num_of_inst();
    const size_t nClass = c.get_class_desc().possible_value_vector().size();

    // Performance tmp:
    double sum_acc = 0.0; //sum of accuracy
    double ave_acc = 0.0; 
    vector<double> sum_trust(nClass,0.0); // sum of trust.
    vector<double> ave_trust(nClass,0.0);
    vector< vector<double> > sum_conf(nClass, vector<double>(nClass,0)); // sum of conf mat.
    vector< vector<double> > ave_conf(nClass, vector<double>(nClass,0));

    for (size_t foldi = 0; foldi<fold(); foldi++) {
	fprintf(stdout, "(I) Cross validating on progress: %d of %d...\n",
		foldi+1, fold());
	c.test_set().clear();
	c.train_set().clear();
	// assign test set.
	c.test_set() = randomIndecs().at(foldi);
	//assign train set.
	c.train_set().resize(nInst-c.test_set().size());
	vector<size_t>::iterator it = c.train_set().begin();
	for (size_t i=0; i<fold(); i++) { // for each fold
	    if (i==foldi) continue;
	    const vector<size_t> & curFold = randomIndecs().at(i);
	    copy( curFold.begin(), curFold.end(), it );
	    it += curFold.size();
	}
	c.train();
	c.test();
	// stores the performance:
	sum_acc += c.accuracy();
	transform ( sum_trust.begin(), sum_trust.end(),
		c.trust().begin(), sum_trust.begin(), double_sum );
	transform ( sum_conf.begin(), sum_conf.end(),
		c.conf().begin(), sum_conf.begin(), vector_sum );
    }
    // average on the performance:
    ave_acc = sum_acc/fold();
    for (size_t i=0;i<nClass;i++) {
	ave_trust.at(i) = sum_trust.at(i) / fold();
    }
    for (size_t i=0;i<nClass;i++) {
	for (size_t j=0;j<nClass;j++) {
	    ave_conf.at(i).at(j) = sum_conf.at(i).at(j) / fold();
	}
    }

#ifdef __XVALIDATOR_DEBUG__
    fprintf(stdout, "(I) The cross validation output:\n");
    fprintf(stdout, "(I) ===========================\n");
    fprintf(stdout, "(I) The average accuracy: %g\n", ave_acc);
    show_conf(c,ave_conf);
    show_trust(c,ave_trust);
#endif
}

void 
Xvalidator::
set_fold(const size_t f)
{
    _fold=f;
    init_randomIndex();
}
