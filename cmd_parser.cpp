//
// Created by Emily & Marc on 10/12/2018.
//

#include "cmd_parser.h"
#include <ctype.h>
#include <algorithm>


const std::string CmdParser::empty_string;

CmdParser::CmdParser(const int argc, char **const argv) {
    bool positional = true;
    for (unsigned int i = 1; i < argc; ++i) {
        std::string tempString(argv[i]);
        if (!tempString.empty() && positional && tempString[0] == '-' && isalpha(tempString[1])) {
            positional = false;
        }
        if (positional) {
            positionalTokens.emplace_back(tempString);
        } else {
            optionalTokens.emplace_back(tempString);
        }
    }
}

const std::string &CmdParser::getPositional(const int position) const {
    if (position >= positionalTokens.size()) return empty_string;
    return positionalTokens[position];
}

unsigned int CmdParser::getPositionalNumber() const {
    return positionalTokens.size();
}

const std::string &CmdParser::getOption(const std::string &option) const {
    std::vector<std::string>::const_iterator itr;
    itr = std::find(this->optionalTokens.begin(), this->optionalTokens.end(), option);
    if (itr != this->optionalTokens.end() && ++itr != this->optionalTokens.end()) {
        return *itr;
    }
    return empty_string;
}

bool CmdParser::optionExists(const std::string &option) const {
    return std::find(this->optionalTokens.begin(), this->optionalTokens.end(), option)
           != this->optionalTokens.end();
}
