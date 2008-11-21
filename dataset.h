/**
 * \file dataset.h
 * \author Kefei Lu
 * \brief Dataset and related class header file.
 *
 * This file contains classes that consists of the Dataset class.
 * \sa dataset.cpp
 */

/**
 * \mainpage Internet Traffic Classificatioin Library Document
 *
 * \author Kefei Lu
 *
 * \section sec_main_intro Introduction
 * 
 * \section sec_main_lib_str Library Structure
 *
 * \section sec_main_lib_usage Library Usage
 * 
 * \section sec_main_references References
 */

#ifndef __DATASET_H__
#define __DATASET_H__

#include "common.h"

using namespace std;

typedef uint64_t NominalType;
typedef double NumericType;

/**
 * \brief The value type of an attribute value.
 *
 * When accessing this type, one must specify the field 
 *   one wants to access, e.g. nom or num. Or the program 
 *   may not interprete the type correctly.
 */
union ValueType {
    NominalType nom;
    NumericType num;
};

/**
 * \brief Attribute Type
 */
typedef enum _AttType {
    ATT_TYPE_NONE = 0, ///< A padding one. 
    ATT_TYPE_NOMINAL,  ///< Nominal type. 
    ATT_TYPE_NUMERIC ///< Numeric type. 
} AttType;

/**
 * \brief Attribute descriptor class.
 *
 * This class is used to describe an attribute. It indicates the type 
 *   of the attribute (AttType), the name of the attribute, and if 
 *   the nominal type, the possible value of the attribut.
 */
class AttDesc {
    private:
	char	name[64];
	AttType	type;
	vector<string> * possibleValues; 
    public:
	// ---- set and get ----
	AttDesc& set_name(const char* name);
	AttDesc& set_type(const AttType _type) {this->type = _type;return *this;};
	AttDesc& set_name_and_type(const char* name, const AttType type);

	/** \brief Clear all the fields. */
	AttDesc& clear();

	/**
	 * \brief Map a between a stored nominal value and its index
	 *
	 * This function maps between a string representing its nominal 
	 *   value and its index in the possibleValue vector.
	 * E.g. {"a", "b", "c"}, map("c") will give 2, map(2) will give
	 *   "c"
	 *
	 * NOTE: It requires the AttDesc to be nominal.
	 */
	size_t map(const string str) const;
	size_t map(const char* str) const;
	string map(const size_t index) const;

	/**
	 * \brief Get a reference to the possible value vector.
	 *
	 * NOTE: It DOES NOT requires the AttDesc to be 
	 *   a nominal one to call. So don't use it on 
	 *   a non-nominal attribute. It makes no sense.
	 */
	vector<string> & possible_value_vector();
	const vector<string> & possible_value_vector() const;


	// ---- c'tor and d'tor ----
	/**
	 * \brief Constructor of the type.
	 *
	 * \param A name describing the att. Take default value ""
	 * \param The type of the attribute.
	 *
	 * The possibleValues ptr will be assigned to a new'd address,
	 *   pointing to a vector of string. It is init to an empty vector.
	 */
	AttDesc(const char* name, const AttType type);

	/** 
	 * \brief Copy constructor.
	 *
	 * It is needed by vector's push_back() methods etc., because this 
	 *   type will need to allocate memory on constructing.
	 *   
	 * \sa AttDesc::operator=(const AttDesc& desc)
	 */
	AttDesc(const AttDesc& desc);

	/**
	 * \brief Copy assignment operator.
	 *
	 * \sa AttDesc(const AttDesc& desc)
	 */
	AttDesc& operator=(const AttDesc& desc);
	
	/** 
	 * \brief Default Destructor
	 *
	 * Will delete the possibleValues ptr is necessary.
	 */
	~AttDesc() {if (possibleValues) delete possibleValues;}; // default d'tor

	/** \brief Return the name of the corresponding Attribute */
	const char* get_name() const
	{
	    return name;
	}

	/** \brief Return the type of the corresponding type */
	AttType	get_type() const
	{
	    return type;
	}
};

/**
 * \brief Attribute instance class.
 *
 * Every Instance is described by an array of attribute values.
 * Attribute can be either a Numeric value or a Nominal value (so far).
 * When an attribute value is unknown, the unknown flag is set to TRUE.
 *
 * \sa Instance, ValueType
 */
class Attribute {
    public:
	ValueType value;
	bool	unknown;
	Attribute() {value.num=0; unknown=0;}
};

/** \brief Instance type.
 *
 * Instance is described by an array of Attributes.
 */
typedef vector<Attribute> Instance;

/**
 * \brief The Dataset class.
 *
 * A dataset consists of:
 *   (1) an array of attribute descriptors (AttDesc), 
 *     which describes each column of Attribute in the corresponding 
 *     instances, for example, the type of the attribute, the possible 
 *     values of the attribute (if it's a nominal one).
 *   (2) a table of instances. 
 *
 * It supports read in an arff file, having only nominal and numeric 
 *   attributes.
 *
 * This is the main class that an classification algorithm should 
 *   operate on. Different algorithms should be described as different 
 *   classes which can take Dataset as an argument, so that the Dataset 
 *   type is acutally made reusable.
 *
 * \sa Instance, Attribute, AttDesc
 */
class Dataset {

    private:
	size_t		_numOfInstance;
	size_t		_numOfAttributes;

	vector<Instance> _inst; ///< the instances in this dataset.
	vector<AttDesc>	_attDesc; ///< describe the instance structure.

	void init(); ///< A private init function for ctor use

    public:
	/** 
	 * \brief Read from arff file.
	 */
	Dataset& read_arff( const char* arff_file );

	/** \brief Init from ARFF file. */
	Dataset(const char* arff_file);

	/** \brief Get the number of instances in this dataset. */
	const size_t num_of_inst() const {return _numOfInstance;}

	/** \brief Get the number of attributes in a instance of this dataset. */
	const size_t num_of_att() const {return _numOfAttributes;}

	/**
	 * \brief Get a reference to the i-th instance.
	 *
	 * By this operator and operator[] of class Instance, 
	 *   dataset[i][j] will return:
	 *   the j-th attribute in the i-th instance
	 *   in the dataset.
	 *
	 * \sa class Instance
	 */
	Instance & operator[] ( const size_t index )
	{
	    assert(index < num_of_inst());
	    return _inst[index];
	}

	const Instance & operator[] ( const size_t index ) const 
	{
	    assert(index < num_of_inst());
	    return _inst[index];
	}

	/** Return a reference to the attribute descriptor vectors (attDesc) */
	AttDesc & get_att_desc(size_t index)
	{
	    assert(index < num_of_att());
	    return _attDesc[index];
	}

	/** A const version of get_att_desc() */
	const AttDesc & get_att_desc(size_t index) const 
	{
	    assert(index < num_of_att());
	    return _attDesc[index];
	}
};

#endif
