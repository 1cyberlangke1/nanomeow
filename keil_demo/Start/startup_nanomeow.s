; nanomeow 最小启动文件（Keil armasm 语法）
; 输入：无。输出：__Vectors 向量表 + Reset_Handler + 栈/堆符号。
; 预期行为：复位后设好 SP，直接跳 __main（microlib 负责清 .bss / 拷 .data，然后调 main）。
; 与 ST 官方 startup 的差别：向量表只保留本固件用得到的 16 项（ST 那份 76 项，多占 240 B）。
; 时钟由 nm_fw.c 的 main() 自己配（HSE x9 = 72 MHz），所以这里不调 SystemInit。
; 堆为 0 字节：本固件全程不用 malloc。

; 栈 1,024 B：引擎的 int64 工作数组已从栈搬到 .bss 的共享暂存，armlink 静态分析给出的最大栈
; 用量是 768 B（搬之前 4,352 B），留 256 B 余量。早先只给 512 B 时栈一溢出就直接踩坏紧随
; 其后的 nm_fb / nm_page / nm_col / logits，板上表现为屏定格 + 串口哑掉（硬 fault）。
Stack_Size      EQU     0x00000400
Heap_Size       EQU     0x00000000

                AREA    STACK, NOINIT, READWRITE, ALIGN=3
                EXPORT  __initial_sp
Stack_Mem       SPACE   Stack_Size
__initial_sp

                AREA    HEAP, NOINIT, READWRITE, ALIGN=3
                EXPORT  __heap_base
                EXPORT  __heap_limit
__heap_base
Heap_Mem        SPACE   Heap_Size
__heap_limit

                AREA    RESET, DATA, READONLY
                EXPORT  __Vectors
                EXPORT  __Vectors_End
                EXPORT  __Vectors_Size

__Vectors       DCD     __initial_sp               ; 0  栈顶
                DCD     Reset_Handler              ; 1  复位
                DCD     0                          ; 2  NMI
                DCD     HardFault_Handler          ; 3  硬 fault
                DCD     0                          ; 4  MemManage
                DCD     0                          ; 5  BusFault
                DCD     0                          ; 6  UsageFault
                DCD     0                          ; 7..10 保留
                DCD     0
                DCD     0
                DCD     0
                DCD     0                          ; 11 SVC
                DCD     0                          ; 12 DebugMon
                DCD     0                          ; 13 保留
                DCD     0                          ; 14 PendSV
                DCD     0                          ; 15 SysTick
__Vectors_End
__Vectors_Size  EQU     __Vectors_End - __Vectors

                AREA    |.text|, CODE, READONLY, ALIGN=2
                THUMB
                PRESERVE8

HardFault_Handler PROC
                EXPORT  HardFault_Handler          [WEAK]
                B       .
                ENDP

Reset_Handler   PROC
                EXPORT  Reset_Handler              [WEAK]
                IMPORT  __main
                LDR     R0, =__main
                BX      R0
                ALIGN   4       ; 字面量池要 4 字节对齐；写显式免得 armasm 自动补字节并报 A1581W
                ENDP

                END
