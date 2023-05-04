# Optimization of GEMM & Conv

```python
alpha = 1000**(-3)
DataTypeBytes = 4
MemoryBandwidth = 651 * (1024 ** 3) #B/s
MaxFLOPS = 13.8 * 1000 # GFLOPS
L2CacheSize = 4718592 #Bytes
# B, Kx, Ky, Nx, Ny, Ni, Nn = 16, 3, 3, 224, 224, 64, 64
```

## Naive Model
```python
# Roofline model of naive way
# For Conv, each block caculate a batchSize of items in output
ConvOperationIntensity = 2*B*Ni*Kx*Ky/(B*Ni*Kx*Ky*2 + B) # 1 for conv1
naiveRoofline = min(MaxFLOPS, ConvOperationIntensity*MemoryBandwidth*alpha) # 698 GFLOPS -> 253 measured
```

## Shared Model
Before we get the naiveRoofline, we find the naive way of using shared memory cannot improve performance. After nvprof using `-m dram_read_throughput`, I find this is because the throughput doesn't increase. I think this is because the naive method has great locality if Strip is 1. Therefore, the cache locality of high and we can assume.
after run `ncu --set full ./conv1` and `ncu --metrics "regex:.*smsp__pcsamp_warps_issue.*" ./conv1` to see the stall reasons.


-------------------------------------
Translated Report (Full Report Below)
-------------------------------------

Process:               nvvp [32740]
Path:                  /private/var/folders/*/nvvp.app/Contents/MacOS/nvvp
Identifier:            com.nvidia.viper.application.product
Version:               12.1.0 (12.1.0.202301050312)
Code Type:             X86-64 (Native)
Parent Process:        launchd [1]
User ID:               501

Date/Time:             2023-05-03 15:27:09.6801 -0700
OS Version:            macOS 13.3.1 (22E261)
Report Version:        12
Bridge OS Version:     7.4 (20P4252)
Anonymous UUID:        AB7BE6CE-BC0E-A09D-A7A7-E660924F6F4D

Sleep/Wake UUID:       4A7EA710-7532-468B-8026-2663D5EAE45C

Time Awake Since Boot: 180000 seconds
Time Since Wake:       6249 seconds

System Integrity Protection: enabled

Crashed Thread:        0  Dispatch queue: com.apple.main-thread

Exception Type:        EXC_BAD_ACCESS (SIGSEGV)
Exception Codes:       KERN_INVALID_ADDRESS at 0x0000000000000008
Exception Codes:       0x0000000000000001, 0x0000000000000008

Termination Reason:    Namespace SIGNAL, Code 11 Segmentation fault: 11
Terminating Process:   exc handler [32740]

VM Region Info: 0x8 is not in any region.  Bytes before following region: 140737486905336
      REGION TYPE                    START - END         [ VSIZE] PRT/MAX SHRMOD  REGION DETAIL
      UNUSED SPACE AT START
--->  
      shared memory            7fffffe9e000-7fffffe9f000 [    4K] r-x/r-x SM=SHM  

Thread 0 Crashed::  Dispatch queue: com.apple.main-thread
0   CoreFoundation                	    0x7ff80d2388e6 _CFGetNonObjCTypeID + 10
1   CoreFoundation                	    0x7ff80d1d1a6d CFBundleGetFunctionPointerForName + 18
2   nvvp                          	       0x10000306b findSymbol + 66
3   nvvp                          	       0x1000017f6 original_main + 1572
4   nvvp                          	       0x100001eb5 main + 1230
5   nvvp                          	       0x100001090 start + 52

Thread 1:
0   libsystem_pthread.dylib       	    0x7ff80d08bbb0 start_wqthread + 0


Thread 0 crashed with X86 Thread State (64-bit):
  rax: 0x000060000020c0a0  rbx: 0x0000000000000000  rcx: 0x6974696e49746573  rdx: 0x0000000000000006
  rdi: 0x0000000000000000  rsi: 0x000060000020c0a0  rbp: 0x00007ff7bfeff8a0  rsp: 0x00007ff7bfeff8a0
   r8: 0x736772416c616974   r9: 0x0000000000000020  r10: 0x000060000020c0a0  r11: 0x00005fff00208947
  r12: 0x00000001000043f8  r13: 0x000060000260c060  r14: 0x000060000020c0a0  r15: 0x0000600000c08720
  rip: 0x00007ff80d2388e6  rfl: 0x0000000000010246  cr2: 0x0000000000000008
  
Logical CPU:     6
Error Code:      0x00000004 (no mapping for user data read)
Trap Number:     14

Thread 0 instruction stream:
  c0 74 0f 48 83 fe 07 75-0a 48 3b 05 12 fe 53 43  .t.H...u.H;...SC
  75 01 c3 48 83 fe 47 77-09 48 8d 05 5a af 7c 41  u..H..Gw.H..Z.|A
  eb 2b 83 c6 b8 8b 05 47-af 7c 41 39 c6 7d e3 89  .+.....G.|A9.}..
  f0 c1 f8 06 48 98 48 8d-0d 35 ae 7c 41 48 8b 84  ....H.H..5.|AH..
  c1 80 00 00 00 48 85 c0-74 c8 83 e6 3f 48 8d 04  .....H..t...?H..
  f0 48 8b 00 c3 90 55 48-89 e5 40 f6 c7 01 75 11  .H....UH..@...u.
 [48]8b 47 08 c1 e8 08 25-ff 03 00 00 e9 87 00 00  H.G....%........	<==
  00 48 8b 05 3a ad da 42-8b 00 31 f8 89 c2 d1 ea  .H..:..B..1.....
  83 e2 07 c1 e8 04 0f b6-c8 48 83 c1 08 83 fa 07  .........H......
  48 0f 45 ca 66 83 f9 06-77 31 b8 16 00 00 00 48  H.E.f...w1.....H
  8d 15 54 00 00 00 48 63-0c 8a 48 01 d1 ff e1 48  ..T...Hc..H....H
  bf ff ff ff ff 01 09 e3-07 e8 d4 10 09 00 84 c0  ................

Binary Images:
       0x100000000 -        0x100003fff com.nvidia.viper.application.product (12.1.0) <33bef1d9-50d6-8117-f7af-49a495a2fb5f> /private/var/folders/*/nvvp.app/Contents/MacOS/nvvp
    0x7ff80d0ef000 -     0x7ff80d58bfef com.apple.CoreFoundation (6.9) <315a3f65-0954-3635-96dc-2f65c691d074> /System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation
    0x7ff80d08a000 -     0x7ff80d095fff libsystem_pthread.dylib (*) <86dfa543-95fa-36b4-83c6-bf03d01b2aad> /usr/lib/system/libsystem_pthread.dylib
               0x0 - 0xffffffffffffffff ??? (*) <00000000-0000-0000-0000-000000000000> ???

External Modification Summary:
  Calls made by other processes targeting this process:
    task_for_pid: 0
    thread_create: 0
    thread_set_state: 0
  Calls made by this process:
    task_for_pid: 0
    thread_create: 0
    thread_set_state: 0
  Calls made by all processes on this machine:
    task_for_pid: 0
    thread_create: 0
    thread_set_state: 0

VM Region Summary:
ReadOnly portion of Libraries: Total=390.4M resident=0K(0%) swapped_out_or_unallocated=390.4M(100%)
Writable regions: Total=557.1M written=0K(0%) resident=0K(0%) swapped_out=0K(0%) unallocated=557.1M(100%)

                                VIRTUAL   REGION 
REGION TYPE                        SIZE    COUNT (non-coalesced) 
===========                     =======  ======= 
Activity Tracing                   256K        1 
Kernel Alloc Once                    8K        1 
MALLOC                           164.1M       16 
MALLOC guard page                   16K        4 
MALLOC_NANO (reserved)           384.0M        1         reserved VM address space (unallocated)
STACK GUARD                       56.0M        2 
Stack                             8712K        2 
VM_ALLOCATE                          8K        2 
__DATA                            14.3M      264 
__DATA_CONST                      13.1M      163 
__DATA_DIRTY                       626K       96 
__FONT_DATA                        2352        1 
__LINKEDIT                       170.7M        2 
__OBJC_RO                         66.2M        1 
__OBJC_RW                         2012K        2 
__TEXT                           219.7M      285 
dyld private memory                260K        2 
mapped file                         52K        1 
shared memory                      568K        6 
===========                     =======  ======= 
TOTAL                              1.1G      852 
TOTAL, minus reserved VM space   716.4M      852 



-----------
Full Report
-----------

{"app_name":"nvvp","timestamp":"2023-05-03 15:27:09.00 -0700","app_version":"12.1.0","slice_uuid":"33bef1d9-50d6-8117-f7af-49a495a2fb5f","build_version":"12.1.0.202301050312","platform":1,"bundleID":"com.nvidia.viper.application.product","share_with_app_devs":1,"is_first_party":0,"bug_type":"309","os_version":"macOS 13.3.1 (22E261)","roots_installed":0,"name":"nvvp","incident_id":"43CC7058-D67B-4287-BF65-4782E774D590"}
{
  "uptime" : 180000,
  "procRole" : "Background",
  "version" : 2,
  "userID" : 501,
  "deployVersion" : 210,
  "modelCode" : "MacBookPro15,4",
  "coalitionID" : 18088,
  "osVersion" : {
    "train" : "macOS 13.3.1",
    "build" : "22E261",
    "releaseType" : "User"
  },
  "captureTime" : "2023-05-03 15:27:09.6801 -0700",
  "incident" : "43CC7058-D67B-4287-BF65-4782E774D590",
  "pid" : 32740,
  "cpuType" : "X86-64",
  "roots_installed" : 0,
  "bug_type" : "309",
  "procLaunch" : "2023-05-03 15:27:07.9434 -0700",
  "procStartAbsTime" : 183894956643465,
  "procExitAbsTime" : 183896692991008,
  "procName" : "nvvp",
  "procPath" : "\/private\/var\/folders\/*\/nvvp.app\/Contents\/MacOS\/nvvp",
  "bundleInfo" : {"CFBundleShortVersionString":"12.1.0","CFBundleVersion":"12.1.0.202301050312","CFBundleIdentifier":"com.nvidia.viper.application.product"},
  "storeInfo" : {"deviceIdentifierForVendor":"6949F767-47A4-5B35-8ABA-45ECBDC52FAB","thirdParty":true},
  "parentProc" : "launchd",
  "parentPid" : 1,
  "coalitionName" : "com.nvidia.viper.application.product",
  "crashReporterKey" : "AB7BE6CE-BC0E-A09D-A7A7-E660924F6F4D",
  "throttleTimeout" : 2147483647,
  "codeSigningID" : "",
  "codeSigningTeamID" : "",
  "codeSigningValidationCategory" : 0,
  "codeSigningTrustLevel" : 0,
  "wakeTime" : 6249,
  "bridgeVersion" : {"build":"20P4252","train":"7.4"},
  "sleepWakeUUID" : "4A7EA710-7532-468B-8026-2663D5EAE45C",
  "sip" : "enabled",
  "vmRegionInfo" : "0x8 is not in any region.  Bytes before following region: 140737486905336\n      REGION TYPE                    START - END         [ VSIZE] PRT\/MAX SHRMOD  REGION DETAIL\n      UNUSED SPACE AT START\n--->  \n      shared memory            7fffffe9e000-7fffffe9f000 [    4K] r-x\/r-x SM=SHM  ",
  "exception" : {"codes":"0x0000000000000001, 0x0000000000000008","rawCodes":[1,8],"type":"EXC_BAD_ACCESS","signal":"SIGSEGV","subtype":"KERN_INVALID_ADDRESS at 0x0000000000000008"},
  "termination" : {"flags":0,"code":11,"namespace":"SIGNAL","indicator":"Segmentation fault: 11","byProc":"exc handler","byPid":32740},
  "vmregioninfo" : "0x8 is not in any region.  Bytes before following region: 140737486905336\n      REGION TYPE                    START - END         [ VSIZE] PRT\/MAX SHRMOD  REGION DETAIL\n      UNUSED SPACE AT START\n--->  \n      shared memory            7fffffe9e000-7fffffe9f000 [    4K] r-x\/r-x SM=SHM  ",
  "extMods" : {"caller":{"thread_create":0,"thread_set_state":0,"task_for_pid":0},"system":{"thread_create":0,"thread_set_state":0,"task_for_pid":0},"targeted":{"thread_create":0,"thread_set_state":0,"task_for_pid":0},"warnings":0},
  "faultingThread" : 0,
  "threads" : [{"triggered":true,"id":2586011,"instructionState":{"instructionStream":{"bytes":[192,116,15,72,131,254,7,117,10,72,59,5,18,254,83,67,117,1,195,72,131,254,71,119,9,72,141,5,90,175,124,65,235,43,131,198,184,139,5,71,175,124,65,57,198,125,227,137,240,193,248,6,72,152,72,141,13,53,174,124,65,72,139,132,193,128,0,0,0,72,133,192,116,200,131,230,63,72,141,4,240,72,139,0,195,144,85,72,137,229,64,246,199,1,117,17,72,139,71,8,193,232,8,37,255,3,0,0,233,135,0,0,0,72,139,5,58,173,218,66,139,0,49,248,137,194,209,234,131,226,7,193,232,4,15,182,200,72,131,193,8,131,250,7,72,15,69,202,102,131,249,6,119,49,184,22,0,0,0,72,141,21,84,0,0,0,72,99,12,138,72,1,209,255,225,72,191,255,255,255,255,1,9,227,7,232,212,16,9,0,132,192],"offset":96}},"threadState":{"r13":{"value":105553156161632},"rax":{"value":105553118412960},"rflags":{"value":66118},"cpu":{"value":6},"r14":{"value":105553118412960},"rsi":{"value":105553118412960},"r8":{"value":8315740862279674228},"cr2":{"value":8},"rdx":{"value":6},"r10":{"value":105553118412960},"r9":{"value":32},"r15":{"value":105553128884000},"rbx":{"value":0},"trap":{"value":14,"description":"(no mapping for user data read)"},"err":{"value":4},"r11":{"value":105548823431495},"rip":{"value":140703349049574,"matchesCrashFrame":1},"rbp":{"value":140702053824672},"rsp":{"value":140702053824672},"r12":{"value":4294984696,"symbolLocation":0,"symbol":"javaVMBundle"},"rcx":{"value":7598814393680553331},"flavor":"x86_THREAD_STATE","rdi":{"value":0}},"queue":"com.apple.main-thread","frames":[{"imageOffset":1349862,"symbol":"_CFGetNonObjCTypeID","symbolLocation":10,"imageIndex":1},{"imageOffset":928365,"symbol":"CFBundleGetFunctionPointerForName","symbolLocation":18,"imageIndex":1},{"imageOffset":12395,"symbol":"findSymbol","symbolLocation":66,"imageIndex":0},{"imageOffset":6134,"symbol":"original_main","symbolLocation":1572,"imageIndex":0},{"imageOffset":7861,"symbol":"main","symbolLocation":1230,"imageIndex":0},{"imageOffset":4240,"symbol":"start","symbolLocation":52,"imageIndex":0}]},{"id":2586085,"frames":[{"imageOffset":7088,"symbol":"start_wqthread","symbolLocation":0,"imageIndex":2}]}],
  "usedImages" : [
  {
    "source" : "P",
    "arch" : "x86_64",
    "base" : 4294967296,
    "CFBundleShortVersionString" : "12.1.0",
    "CFBundleIdentifier" : "com.nvidia.viper.application.product",
    "size" : 16384,
    "uuid" : "33bef1d9-50d6-8117-f7af-49a495a2fb5f",
    "path" : "\/private\/var\/folders\/*\/nvvp.app\/Contents\/MacOS\/nvvp",
    "name" : "nvvp",
    "CFBundleVersion" : "12.1.0.202301050312"
  },
  {
    "source" : "P",
    "arch" : "x86_64h",
    "base" : 140703347699712,
    "CFBundleShortVersionString" : "6.9",
    "CFBundleIdentifier" : "com.apple.CoreFoundation",
    "size" : 4837360,
    "uuid" : "315a3f65-0954-3635-96dc-2f65c691d074",
    "path" : "\/System\/Library\/Frameworks\/CoreFoundation.framework\/Versions\/A\/CoreFoundation",
    "name" : "CoreFoundation",
    "CFBundleVersion" : "1971"
  },
  {
    "source" : "P",
    "arch" : "x86_64",
    "base" : 140703347286016,
    "size" : 49152,
    "uuid" : "86dfa543-95fa-36b4-83c6-bf03d01b2aad",
    "path" : "\/usr\/lib\/system\/libsystem_pthread.dylib",
    "name" : "libsystem_pthread.dylib"
  },
  {
    "size" : 0,
    "source" : "A",
    "base" : 0,
    "uuid" : "00000000-0000-0000-0000-000000000000"
  }
],
  "sharedCache" : {
  "base" : 140703343149056,
  "size" : 21474836480,
  "uuid" : "1b64bf32-be7f-304b-add0-ce61655e2402"
},
  "vmSummary" : "ReadOnly portion of Libraries: Total=390.4M resident=0K(0%) swapped_out_or_unallocated=390.4M(100%)\nWritable regions: Total=557.1M written=0K(0%) resident=0K(0%) swapped_out=0K(0%) unallocated=557.1M(100%)\n\n                                VIRTUAL   REGION \nREGION TYPE                        SIZE    COUNT (non-coalesced) \n===========                     =======  ======= \nActivity Tracing                   256K        1 \nKernel Alloc Once                    8K        1 \nMALLOC                           164.1M       16 \nMALLOC guard page                   16K        4 \nMALLOC_NANO (reserved)           384.0M        1         reserved VM address space (unallocated)\nSTACK GUARD                       56.0M        2 \nStack                             8712K        2 \nVM_ALLOCATE                          8K        2 \n__DATA                            14.3M      264 \n__DATA_CONST                      13.1M      163 \n__DATA_DIRTY                       626K       96 \n__FONT_DATA                        2352        1 \n__LINKEDIT                       170.7M        2 \n__OBJC_RO                         66.2M        1 \n__OBJC_RW                         2012K        2 \n__TEXT                           219.7M      285 \ndyld private memory                260K        2 \nmapped file                         52K        1 \nshared memory                      568K        6 \n===========                     =======  ======= \nTOTAL                              1.1G      852 \nTOTAL, minus reserved VM space   716.4M      852 \n",
  "legacyInfo" : {
  "threadTriggered" : {
    "queue" : "com.apple.main-thread"
  }
},
  "logWritingSignature" : "e43831fe73572acf432f1878f3225c3b2944e0d3",
  "trialInfo" : {
  "rollouts" : [
    {
      "rolloutId" : "60f8ddccefea4203d95cbeef",
      "factorPackIds" : {

      },
      "deploymentId" : 240000025
    },
    {
      "rolloutId" : "5fb4245a1bbfe8005e33a1e1",
      "factorPackIds" : {

      },
      "deploymentId" : 240000021
    }
  ],
  "experiments" : [
    {
      "treatmentId" : "c28e4ee6-1b08-4f90-8e05-2809e78310a3",
      "experimentId" : "6317d2003d24842ff850182a",
      "deploymentId" : 400000013
    },
    {
      "treatmentId" : "6dd670af-0633-45e4-ae5f-122ae4df02be",
      "experimentId" : "64406ba83deb637ac8a04419",
      "deploymentId" : 900000005
    }
  ]
}
}

Model: MacBookPro15,4, BootROM 1968.100.17.0.0 (iBridge: 20.16.4252.0.0,0), 4 processors, Quad-Core Intel Core i5, 1.4 GHz, 16 GB, SMC 
Graphics: Intel Iris Plus Graphics 645, Intel Iris Plus Graphics 645, Built-In
Display: Color LCD, 2560 x 1600 Retina, MirrorOff, Online
Display: DVI, 1920 x 1080 (1080p FHD - Full High Definition), Main, MirrorOff, Online
Memory Module: BANK 0/ChannelA-DIMM0, 8 GB, LPDDR3, 2133 MHz, Micron, MT52L1G32D4PG-093
Memory Module: BANK 2/ChannelB-DIMM0, 8 GB, LPDDR3, 2133 MHz, Micron, MT52L1G32D4PG-093
AirPort: spairport_wireless_card_type_wifi (0x14E4, 0x870), wl0: Jan 14 2023 01:22:18 version 16.20.357.3.3.6.118 FWID 01-eaad3d99
Bluetooth: Version (null), 0 services, 0 devices, 0 incoming serial ports
Network Service: Wi-Fi, AirPort, en0
USB Device: USB3.0 Hub
USB Device: USB31Bus
USB Device: USB2.0 Hub
USB Device: T2Bus
USB Device: Touch Bar Backlight
USB Device: Touch Bar Display
USB Device: Apple Internal Keyboard / Trackpad
USB Device: Headset
USB Device: Ambient Light Sensor
USB Device: FaceTime HD Camera (Built-in)
USB Device: Apple T2 Controller
Thunderbolt Bus: MacBook Pro, Apple Inc., 63.5
