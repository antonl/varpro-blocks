#pragma once

#include <string>
#include <armadillo>
#include <tuple>

using std::string;
using arma::mat;
using arma::umat;
using arma::vec;
using arma::uword;

template <typename T> std::string get_size(const T);

class response_block 
{
    public:
        explicit response_block(const vec& measured);
        virtual ~response_block() = default;

        // deleted copy ctor and copy assignment
        response_block(const response_block& other) = delete;
        response_block& operator=(const response_block& other) = delete;

        void update_model(const vec p, bool update_jac=false);
        virtual const string get_name() const;

        //vec full_jacobian() const;

        virtual void _generate_model(const vec& p);
        virtual void _generate_jacobian(const vec& p);

    public:
        const vec y; // measured response
        vec yh; // estimated response
        vec resid; // residuals

        uword M; // number of measurements
        uword P; // number of nonlinear parameters
        uword N; // number of linear parameters

        vec alpha; // nonlinear parameter vector
        vec beta; // linear parameter vector

        mat Amat; // model matrix
        
        umat jidx; // indexing matrix for sparse jacobian
        mat mjac; // sparse matrix jacobian
};

class single_exp_model : public response_block
{
    public:
        const vec tvec;

    public:
        explicit single_exp_model(const vec& m, const vec& t);
        virtual ~single_exp_model() = default;

        virtual const string get_name() const;
        
        const vec& get_t() const;

        virtual void _generate_model(const vec& p);
        virtual void _generate_jacobian(const vec& p);
        
        // deleted copy ctor and copy assignment
        //single_exp_model(const single_exp_model& other) = delete;
        //single_exp_model& operator=(const single_exp_model& other) = delete;
};
