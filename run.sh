# !/bin/bash

package="parallel-scan-sky"

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "$package - attempt to capture frames"
      echo " "
      echo "$package [options] application [arguments]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-a, --action=ACTION       specify an action to use"
      echo "-o, --output-dir=DIR      specify a directory to store output in"
      exit 0
      ;;
    -sequential)
      shift
      if test $1; then
              sh ./sequential/src/run.sh $1
      else
              echo "no input file specified"
              exit 1
      fi
      shift
      ;;
    -openmp)
        shift
        if test $1; then
                sh ./openmp/src/run.sh $1
        else
                echo "no input file specified"
                exit 1
        fi
        shift
        ;;
    *)
      break
      ;;
  esac
done
