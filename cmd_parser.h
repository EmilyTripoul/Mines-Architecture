//
// Created by Emily & Marc on 10/12/2018.
//

#pragma once

#include <string>
#include <vector>
#include <sstream>

class CmdParser{
public:
    CmdParser (const int argc, char ** const argv );

    const std::string& getPositional(const int position) const;
    template<class T>
    const T getPositionalAs(const int position) const;
    unsigned int getPositionalNumber() const;

    const std::string& getOption(const std::string &option) const;
    template<class T>
    const T getOptionAs(const std::string &option) const;
    bool optionExists(const std::string &option) const;

private:
    std::vector <std::string> positionalTokens;
    std::vector <std::string> optionalTokens;
    static const std::string empty_string;
};


template<class T>
const T CmdParser::getPositionalAs(const int position) const {
    T result;
    std::stringstream sstream(getPositional(position));
    sstream>>result;
    return result;
}

template<class T>
const T CmdParser::getOptionAs(const std::string &option) const {
    T result;
    std::stringstream sstream(getOption(option));
    sstream>>result;
    return result;
}