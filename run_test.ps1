$fileExe = ".\cmake-build-release\Mines_Archi.exe"
$fileExeAligned = ".\cmake-build-release\Mines_Archi_aligned.exe"

#& $fileExe -h
#& $fileExe 6 0 0.5243
& $fileExe 10000000 150 0 -run 10
& $fileExeAligned 10000000 150 0 -run 10
