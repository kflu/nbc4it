/**
 * \file xvalidator.h
 * \author Kefei Lu
 * \brief Class structure description on cross validation for classifiers.
 */
#ifndef __XVALIDATOR_H__
#define __XVALIDATOR_H__
#include "common.h"
#include "dataset.h"
#include "classifier.h"

/**
 * The cross validation structure.
 *
 */
class Xvalidator {
    private:
	RSeed		_seed;
	Classifier *	_binded_classifier;
	size_t		_fold;

	/** Randomized instance indecs. 
	 *
	 * It must have _fold vectors, each stores a portion (nInst/_fold, 
	 * except the last one.) of instance indecs, unique from the others.*/
	vector< vector<size_t> > _randomIndecs;
    public:
	RSeed& seed() {return _seed;}
	const RSeed& seed() const {return _seed;}
	Classifier & classifier() const {return *_binded_classifier;}
	const size_t & fold() const {return _fold;}
	void set_fold(const size_t f) {_fold = f;}

	Xvalidator(Classifier* c, const size_t fold = 3, RSeed seed=0);

	vector< vector<size_t> >& randomIndecs() {return _randomIndecs;}
	const vector< vector<size_t> >& randomIndecs() const {return _randomIndecs;}
	/** Initialize _randomIndecs.
	 *
	 * It must have _fold vectors, each stores a portion (nInst/_fold, 
	 * except the last one.) of instance indecs, unique from the others.
	 *
	 * \sa _randomIndecs */
	void init_randomIndex();

	/** Randomize the randomIndecs.
	 *
	 * It puts (sorted) randomized indecs into the _randomIndecs vectors.
	 * It must be done before the whole cross validation 
	 * process, but NOT during the process. */
	void randomize();

	void xvalidate();
};

#endif
