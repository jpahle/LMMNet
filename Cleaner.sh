#PBS -N CLEANER
#PBS -V
#PBS -o log/CLEANER.log
#PBS -l cput=1000:00:00

Rscript --verbose /net/home.isilon/ag-pahle/Arne/PHD/Cleaner.R > /net/home.isilon/ag-pahle/Arne/PHD/Cleaner.Rout 2>&1
