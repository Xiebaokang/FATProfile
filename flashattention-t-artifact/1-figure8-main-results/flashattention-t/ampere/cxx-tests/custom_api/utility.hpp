#pragma once

#include <string>
#include <sstream>


#include <filesystem>
#include <cstdio>
#include <unistd.h>
#include <cstring>



// Inline function to colorize text
inline std::string color_text(const std::string& text, const std::string& color_code) {
    return "\033[" + color_code + "m" + text + "\033[0m";
}

// Helper functions for specific colors
inline std::string red_text(const std::string& text) {
    return color_text(text, "31");
}

inline std::string green_text(const std::string& text) {
    return color_text(text, "32");
}

inline std::string yellow_text(const std::string& text) {
    return color_text(text, "33");
}


// template <typename Func>
// void intercept_stdout_tofile(Func const& func, std::filesystem::path const& filepath){
//     fflush(stdout);
//     int stdout_fd = dup(STDOUT_FILENO);
//     FILE* const file_ptr = fopen(filepath.c_str(), "w");
//     int redir_fd = fileno(file_ptr);
//     dup2(redir_fd, STDOUT_FILENO);
//     func();
//     fflush(stdout);
//     fclose(file_ptr);
//     dup2(stdout_fd, STDOUT_FILENO);
//     close(stdout_fd);
// }

// template <typename Func, size_t Bufsize = 4096>
// std::string intercept_stdout_tostr(Func const& func){
//     fflush(stdout);
//     int stdout_fd = dup(STDOUT_FILENO);
//     int pipefd[2];
//     pipe(pipefd);
//     dup2(pipefd[1], STDOUT_FILENO);
//     func();
//     fflush(stdout);
//     close(pipefd[1]);
//     char* buffer = (char*)malloc(Bufsize);
//     memset(buffer, 0, Bufsize);
//     std::string result;
//     while (1) {
//         int readsize = Bufsize-1; // leave one byte for null terminator
//         int bytes = read(pipefd[0], buffer, readsize);
//         result += std::string{buffer};
//         if (bytes < readsize){
//             break;
//         }
//     }
//     close(pipefd[0]);
//     dup2(stdout_fd, STDOUT_FILENO);
//     close(stdout_fd);
//     return result;
// }