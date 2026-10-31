# Tests configuration

include(FetchContent)

# Fetch Catch2
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.2
)
FetchContent_MakeAvailable(Catch2)

# Test executable
add_executable(wake_word_tests
    test_main.cpp
    test_audio.cpp
    test_recorder.cpp
)

target_link_libraries(wake_word_tests PRIVATE
    wake_word_lib
    Catch2::Catch2WithMain
)

# Register tests with CTest
include(CTest)
include(Catch)
catch_discover_tests(wake_word_tests)
