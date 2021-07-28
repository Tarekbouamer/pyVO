#!/usr/bin/env bash

# ====================================================

echo '================================================'
echo "Building Thirdparty"
echo '================================================'

STARTING_DIR=`pwd`  # this should be the main folder directory of the repo

if [[ -n "$EXTERNAL_OPTION" ]]; then
    echo "external option: $EXTERNAL_OPTION" 
fi

# check if we want to add a python interpreter check
if [[ -n "$WITH_PYTHON_INTERP_CHECK" ]]; then
    echo "WITH_PYTHON_INTERP_CHECK: $WITH_PYTHON_INTERP_CHECK " 
    EXTERNAL_OPTION="$EXTERNAL_OPTION -DWITH_PYTHON_INTERP_CHECK=$WITH_PYTHON_INTERP_CHECK"
fi
# ====================================================

CURRENT_USED_PYENV=$(get_virtualenv_name)
echo "currently used pyenv: $CURRENT_USED_PYENV"

echo '================================================'
echo "Configuring and building thirdparty/Pangolin ..."


INSTALL_PANGOLIN_ORIGINAL=0
cd third_party
if [ $INSTALL_PANGOLIN_ORIGINAL -eq 1 ] ; then
    # N.B.: pay attention this will generate a module 'pypangolin' ( it does not have the methods dcam.SetBounds(...) and pangolin.DrawPoints(points, colors)  )
    if [ ! -d pangolin ]; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get install -y libglew-dev
        fi     
        git clone https://github.com/stevenlovegrove/Pangolin.git pangolin
        cd pangolin
        git submodule init && git submodule update
        cd ..
    fi
    cd pangolin
    make_dir build 
    if [ ! -f build/src/libpangolin.so ]; then
        cd build
        cmake ../ -DAVFORMAT_INCLUDE_DIR="" -DCPP11_NO_BOOST=ON $EXTERNAL_OPTION
        make -j8
        cd build/src
        ln -s pypangolin.*-linux-gnu.so  pangolin.linux-gnu.so
    fi
else
    # N.B.: pay attention this will generate a module 'pangolin' 
    if [ ! -d pangolin ]; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then    
            sudo apt-get install -y libglew-dev
            git clone https://github.com/uoip/pangolin.git
            cd pangolin
            PANGOLIN_UOIP_REVISION=3ac794a
            git checkout $PANGOLIN_UOIP_REVISION
            cd ..      
            # copy local changes 
            rsync ./pangolin_changes/python_CMakeLists.txt ./pangolin/python/CMakeLists.txt             
        fi 
        if [[ "$OSTYPE" == "darwin"* ]]; then
            git clone --recursive https://gitlab.com/luigifreda/pypangolin.git pangolin 
        fi 
    fi
    cd pangolin
    if [ ! -f pangolin.cpython-*.so ]; then   
        make_dir build   
        cd build
        cmake .. -DBUILD_PANGOLIN_LIBREALSENSE=OFF $EXTERNAL_OPTION # disable realsense 
        make -j8
        cd ..
        #python setup.py install
    fi
fi
cd $STARTING_DIR

