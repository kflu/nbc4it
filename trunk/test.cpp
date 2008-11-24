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

using namespace std;

int main()
{
    //stdout = stderr;

    Dataset dataset("test.arff");

    NaiveBayesClassifier c(dataset,248);
    // This is for testing attribute distribution correctness.
    // ----------
    //c.init_tt_set();
    //c.train();
    //c.test();
    //size_t nAtt = c.dataset().num_of_att();
    //for (size_t i=0;i<nAtt;i++) {
    //    printf("%d -th attribute:",i);
    //    if (i==c.class_index()) continue;
    //    const Distribution * distr = c.attDistrOnClass().table()[1][i];
    //    if (c.dataset().get_att_desc(i).get_type() == ATT_TYPE_NOMINAL) {
    //        for (size_t j=0;j<c.dataset().get_att_desc(i).possible_value_vector().size();j++) {
    //    	printf("%g ",static_cast<const NominalDistribution*>(distr)->pmf().at(j));
    //        }
    //        printf("\n");
    //    }
    //    else if (c.dataset().get_att_desc(i).get_type() == ATT_TYPE_NUMERIC) {
    //        printf("mean: %g, var: %g, (invalid: %d)\n",
    //    	    static_cast<const NormalDistribution*>(distr)->mean(),
    //    	    static_cast<const NormalDistribution*>(distr)->var(),
    //    	    static_cast<const NormalDistribution*>(distr)->invalid());
    //    }
    //}
    // ----------

    // Cross validation:
    // ----------------------
    Xvalidator x(&c);
    x.xvalidate();
    x.set_fold(5);
    


    return 0;
}
