/**
 * \file dataset.cpp
 * \author Kefei Lu
 * \brief The implementations of class and methods declared in dataset.h
 * \sa dataset.h
 */

#include "dataset.h"
using namespace std;

#define MAX_LINE_CHAR 20000
//#define __DATASET_DEBUG__

AttDesc& 
AttDesc::set_name(const char* name)
{
    strncpy(this->name, name, sizeof(this->name));
    return *this;
}

AttDesc& 
AttDesc::set_name_and_type(const char* name, const AttType type)
{
    set_name(name);
    set_type(type);
    return *this;
}

vector<string> & 
AttDesc::possible_value_vector()
{
    return *possibleValues;
}

const vector<string> & 
AttDesc::possible_value_vector() const
{
    return *possibleValues;
}

AttDesc&
AttDesc::clear()
{
    strcpy(name, "");
    type = ATT_TYPE_NONE;
    possibleValues->clear();
    return *this;
}

size_t 
AttDesc::map(const string str) const
{
    assert(type==ATT_TYPE_NOMINAL);
    vector<string>::iterator it;
    for ( it=(*possibleValues).begin(); it <(*possibleValues).end(); it++)
    {
	if (*it == str) return it -(*possibleValues).begin();
    }
    fprintf(stderr, "(E) Invalid possible value: %s\n", str.c_str());
    exit(1);
}

size_t
AttDesc::map(const char* str) const
{
    assert(type==ATT_TYPE_NOMINAL);
    return map(string(str));
}

string
AttDesc::map(const size_t index) const
{
    assert(type==ATT_TYPE_NOMINAL);
    return (*possibleValues)[index];
}

AttDesc::AttDesc(const char* name = "", const AttType type = ATT_TYPE_NONE)
{
    set_name_and_type(name,type);
    possibleValues = new vector<string>;
    possibleValues->clear();
}

AttDesc::AttDesc(const AttDesc& desc)
{
    set_name_and_type(desc.get_name(),desc.get_type());
    possibleValues = new vector<string>;
    for (size_t i=0; i<desc.possible_value_vector().size(); i++) {
	(*possibleValues).push_back( (desc.possible_value_vector())[i] );
    }
}

AttDesc&
AttDesc::operator=(const AttDesc& desc)
{
    set_name_and_type(desc.get_name(), desc.get_type());
    for (size_t i=0; i<desc.possible_value_vector().size(); i++) {
	(*possibleValues).push_back( (desc.possible_value_vector())[i] );
    }
    return *this;
}

void
Dataset::init()
{
    _numOfInstance = 0;
    _numOfAttributes = 0;
    _inst.clear();
    _attDesc.clear();
    return;
}

Dataset& 
Dataset::read_arff( const char* arff_file )
{
    fprintf( stdout, "(I) Opening file: %s...\n", arff_file );
    FILE* arff = fopen( arff_file, "r" );
    if (!arff) {
	fprintf(stderr, "(E) Opening file %s failed.\n", arff_file);
	exit(1);
    }

    init();

    AttDesc desc;
    Instance inst;
    char buf[MAX_LINE_CHAR];
    char* result = NULL;
    short int flag_data_begin = 0;

    fprintf( stdout, "(I) Loading attributes and instances...\n" );
    while (fgets(buf, MAX_LINE_CHAR, arff) != NULL) {
	if (!flag_data_begin) {
	    // still looking for @command
	    desc.clear();
	    // Parse the first word.
	    result = strtok(buf, " \n");
	    if (result == NULL) continue;
	    if (strcmp(result, "@attribute") == 0) {
		result = strtok(NULL, " \n"); // name
		assert(result);
		desc.set_name(result);
		result = strtok(NULL, " \n"); // type
		assert(result);

		if (strcmp(result,"numeric")==0) {
		    // Numeric type
		    desc.set_type(ATT_TYPE_NUMERIC);
#ifdef __DATASET_DEBUG__
		    cout << desc.get_name() << " " << desc.get_type() << " ";
		    cout << endl;
#endif
		    _attDesc.push_back(desc);
		}
		else if (result[0] == '{') {
		    // Nominal type
		    desc.set_type(ATT_TYPE_NOMINAL);
#ifdef __DATASET_DEBUG__
		    cout << desc.get_name() << " " << desc.get_type() << " ";
#endif
		    char * tmp = result;
		    while((result = strtok(tmp, "{, }\n"))!=NULL) {
			tmp = NULL;
			// read all possible values
#ifdef __DATASET_DEBUG__
			cout << result << " ";
#endif
			desc.possible_value_vector().push_back(result);
		    }
#ifdef __DATASET_DEBUG__
		    cout << endl;
#endif
		    _attDesc.push_back(desc);
		}
	    }
	    else if (strcmp(result, "@data") == 0) {
		// End of Attribute desc, Begin of dataset
		flag_data_begin = 1;
	    }
	}
	else {
	    inst.clear();
	    char* buftmp = new char [MAX_LINE_CHAR];
	    char * buftmp_orig = buftmp;
	    strcpy(buftmp, buf);
	    // we are now in data section. get instances.
	    while ((result = strtok(buftmp, ", \n"))!=NULL) {

		buftmp = NULL; // because following strtok call must have NULL str.
		size_t i = inst.size();
		Attribute tmpatt;
		// first check the corresponding attDesc,
		if ( _attDesc[i].get_type() == ATT_TYPE_NUMERIC ) {
		    //   if numeric then string -> double, store.
		    if ( strcmp(result, "?") == 0 ) {
			// Attribute unknown
			tmpatt.unknown = 1;
		    } else {
			char * tailptr = NULL;
			tmpatt.value.num = NumericType(strtod(result, &tailptr));
			if (tailptr == result) {
			    fprintf(stderr, "(E) Processing invalue numeric value: %s\n", result);
			    exit(1);
			}
		    }
		    inst.push_back(tmpatt);

		} else if ( _attDesc[i].get_type() == ATT_TYPE_NOMINAL ) {
		    //   if nominal then string -> index of possible values, store.
		    if ( strcmp(result, "?") == 0 ) {
			tmpatt.unknown = 1;
		    } else {
			tmpatt.value.nom = NominalType(_attDesc[i].map(result));
		    }
		    inst.push_back(tmpatt);

		} else {
		    fprintf(stderr, "(E) Type must be either numeric or nominal.\n");
		    exit(1);
		}
	    } // reading data section
	    delete [] buftmp_orig;
	    // Check if inst have same numOfAtt as in _attDesc:
	    assert(inst.size() == _attDesc.size());
	    _inst.push_back(inst);
	}
    } // read every line into buf
    // Finalize
    _numOfAttributes = _attDesc.size();
    _numOfInstance = _inst.size();

    fprintf( stdout, "(I) Read %d attributes, %d instances.\n", 
	    _numOfAttributes, _numOfInstance );

    fclose(arff);
    fprintf( stdout, "(I) File %s closed.\n", arff_file );

    return *this;
}

Dataset::Dataset(const char* arff_file)
{
    read_arff(arff_file);
}

