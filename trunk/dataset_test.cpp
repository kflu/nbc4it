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

using namespace std;

int main()
{
    stdout = stderr;

    Dataset dataset("test.arff");

    NaiveBayesClassifierFake c;
    c.bind_dataset(dataset);
    c.class_index() = 248;

    //c.randomize();
    //c.ran_tt_set();
    c.init_tt_set();
    c.train();
    c.test();

    /*
    size_t nInst = dataset.num_of_inst();
    size_t nAtt = dataset.num_of_att();
    for (size_t i=0; i<nInst; i++) {
	for (size_t k=0; k<nAtt; k++) {
	    if (dataset.get_att_desc(k).get_type() == ATT_TYPE_NOMINAL) {
		if (dataset[i][k].unknown) {
		    fprintf(stdout, "?");
		} else {
		    fprintf(stdout, "%s", dataset.get_att_desc(k).map(dataset[i][k].value.nom).c_str());
		}
		fprintf(stdout, ",");
	    } else if (dataset.get_att_desc(k).get_type() == ATT_TYPE_NUMERIC) {
		if (dataset[i][k].unknown) {
		    fprintf(stdout, "?");
		} else {
		    fprintf(stdout, "%g",dataset[i][k].value.num);
		}
		fprintf(stdout, ",");
	    }
	}
	fprintf(stdout, "\n");
    }
    */
    return 0;
}
