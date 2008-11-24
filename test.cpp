/**
 * \file dataset_test.cpp
 * \author Kefei Lu
 * \brief A test on dataset reading from arff file.
 * \sa dataset.h dataseet.cpp
 */
#include "common.h"
#include <iostream>
#include "dataset.h"
#include "classifier.h"
#include "xvalidator.h"

/** Only use the attributes which are proved to be more important. */
#define __ONLY_USE_THESE_ATT__

using namespace std;

int main()
{
    Dataset dataset("test.arff");

    NaiveBayesClassifier c(dataset,248);

#ifdef __ONLY_USE_THESE_ATT__
    /* Only use the attributes which are proved to be more important. */
    // size_t use_these[] = {1};
     size_t use_these[] = {1,60, 95, 96, 86, 162, 45, 180, 83, 113, 59};
    // size_t use_these[] = {60, 95, 96, 86, 162, 45, 180, 83, 113, 59};
    c.only_these_att().assign(use_these, use_these + sizeof(use_these)/sizeof(size_t));
    c.useAllAtt() = 0;
#endif

    // Cross validation:
    Xvalidator x(&c);
    // x.seed() = 2;
    x.set_fold(8);
    x.xvalidate();

#ifdef __TEST_DEBUG__
    // This is for testing attribute distribution correctness.
    // ----------
    c.init_tt_set();
    c.train();
    c.test();
    size_t nAtt = c.dataset().num_of_att();
    for (size_t i=0;i<nAtt;i++) {
        printf("%d -th attribute:",i);
        if (i==c.class_index()) continue;
        const Distribution * distr = c.attDistrOnClass().table()[1][i];
        if (c.dataset().get_att_desc(i).get_type() == ATT_TYPE_NOMINAL) {
            for (size_t j=0;j<c.dataset().get_att_desc(i).possible_value_vector().size();j++) {
        	printf("%g ",static_cast<const NominalDistribution*>(distr)->pmf().at(j));
            }
            printf("\n");
        }
        else if (c.dataset().get_att_desc(i).get_type() == ATT_TYPE_NUMERIC) {
            printf("mean: %g, var: %g, (invalid: %d)\n",
        	    static_cast<const NormalDistribution*>(distr)->mean(),
        	    static_cast<const NormalDistribution*>(distr)->var(),
        	    static_cast<const NormalDistribution*>(distr)->invalid());
        }
    }
    // ----------
#endif
    return 0;
}
