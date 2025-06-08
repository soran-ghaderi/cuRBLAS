# Documentation CMakeLists.txt
find_package(Doxygen REQUIRED)

if(DOXYGEN_FOUND)
    # Configure Doxygen variables
    set(DOXYGEN_PROJECT_NAME "cuRBLAS")
    set(DOXYGEN_PROJECT_NUMBER "0.0.1")
    set(DOXYGEN_PROJECT_BRIEF "CUDA Randomized BLAS Library")
    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/api/doxygen")
    set(DOXYGEN_INPUT "${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/README.md")
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_SOURCE_DIR}/README.md")
    
    # Ensure output directory exists
    file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIRECTORY})
    
    # Create a custom Doxyfile with proper settings
    set(DOXYFILE_CONTENT "
PROJECT_NAME           = \"${DOXYGEN_PROJECT_NAME}\"
PROJECT_NUMBER         = \"${DOXYGEN_PROJECT_NUMBER}\"
PROJECT_BRIEF          = \"${DOXYGEN_PROJECT_BRIEF}\"
OUTPUT_DIRECTORY       = ${DOXYGEN_OUTPUT_DIRECTORY}
INPUT                  = ${DOXYGEN_INPUT}
USE_MDFILE_AS_MAINPAGE = ${DOXYGEN_USE_MDFILE_AS_MAINPAGE}
RECURSIVE              = YES
GENERATE_HTML          = YES
GENERATE_LATEX         = NO
EXTRACT_ALL            = YES
SOURCE_BROWSER         = YES
INLINE_SOURCES         = NO
STRIP_CODE_COMMENTS    = YES
OPTIMIZE_OUTPUT_FOR_C  = YES
EXTENSION_MAPPING      = .cu=C++ .cuh=C++
FILE_PATTERNS          = *.c *.cpp *.cxx *.h *.hpp *.hxx *.cu *.cuh *.md
EXCLUDE                = build/ cmake-build-debug/ .git/ .idea/ site/ .venv/
")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile ${DOXYFILE_CONTENT})

    # Note the option ALL which allows to build the docs together with the application
    add_custom_target(docs ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)

    # Optional: Also create a target for just building docs without ALL
    add_custom_target(doxygen
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
        
else (DOXYGEN_FOUND)
    message("Doxygen needs to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)


