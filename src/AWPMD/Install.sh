# Install/unInstall package files in LAMMPS
# mode = 0/1/2 for uninstall/install/update

mode=$1

# arg1 = file, arg2 = file it depends on

# enforce using portable C locale
LC_ALL=C
export LC_ALL

cat <<EOF
WARNING-WARNING-WARNING-WARNING-WARNING-WARNING-WARNING-WARNING-WARNING-WARNING
WARNING-WARNING-WARNING-WARNING-WARNING-WARNING-WARNING-WARNING-WARNING-WARNING

 The AWPMD package will be removed from LAMMPS in Summer 2025 due to lack of
 maintenance and use of code constructs that conflict with modern C++ compilers
 and standards.  Please contact developers@lammps.org if you have any concerns
 about this step.

WARNING-WARNING-WARNING-WARNING-WARNING-WARNING-WARNING-WARNING-WARNING-WARNING
WARNING-WARNING-WARNING-WARNING-WARNING-WARNING-WARNING-WARNING-WARNING-WARNING
EOF

action () {
  if (test $mode = 0) then
    rm -f ../$1
  elif (! cmp -s $1 ../$1) then
    if (test -z "$2" || test -e ../$2) then
      cp $1 ..
      if (test $mode = 2) then
        echo "  updating src/$1"
      fi
    fi
  elif (test -n "$2") then
    if (test ! -e ../$2) then
      rm -f ../$1
    fi
  fi
}

# all package files with no dependencies

for file in *.cpp *.h; do
  test -f ${file} && action $file
done

# edit 2 Makefile.package files to include/exclude package info

if (test $1 = 1) then

  if (test -e ../Makefile.package) then
    sed -i -e 's/[^ \t]*awpmd[^ \t]* //g' ../Makefile.package
    sed -i -e 's|^PKG_INC =[ \t]*|&-I../../lib/awpmd/ivutils/include -I../../lib/awpmd/systems/interact |' ../Makefile.package
    sed -i -e 's|^PKG_PATH =[ \t]*|&-L../../lib/awpmd |' ../Makefile.package
    sed -i -e 's|^PKG_LIB =[ \t]*|&-lawpmd |' ../Makefile.package
    sed -i -e 's|^PKG_SYSPATH =[ \t]*|&$(awpmd_SYSPATH) |' ../Makefile.package
    sed -i -e 's|^PKG_SYSLIB =[ \t]*|&$(awpmd_SYSLIB) |' ../Makefile.package
  fi

  if (test -e ../Makefile.package.settings) then
    sed -i -e '/^[ \t]*include.*awpmd.*$/d' ../Makefile.package.settings
    # multiline form needed for BSD sed on Macs
    sed -i -e '4 i \
include ..\/..\/lib\/awpmd\/Makefile.lammps
' ../Makefile.package.settings
  fi

elif (test $1 = 0) then

  if (test -e ../Makefile.package) then
    sed -i -e 's/[^ \t]*awpmd[^ \t]* //g' ../Makefile.package
  fi

  if (test -e ../Makefile.package.settings) then
    sed -i -e '/^[ \t]*include.*awpmd.*$/d' ../Makefile.package.settings
  fi

fi
