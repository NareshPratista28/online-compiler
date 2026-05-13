package resh12_gmail_com; 

public class HitungTotalSaldo {
    public static void main(String[] args) {
        double saldoAwal = 1000000.0; // Inisialisasi dengan 1000000.0
        int jumlahDeposito = 5;
        int jumlahPenarikan = 2;

        double totalSaldo; // Deklarasi variabel

        totalSaldo = saldoAwal + (jumlahDeposito * 10000) - jumlahPenarikan;

        System.out.println("Detail Transaksi:");
        System.out.println("Saldo awal: Rp " + saldoAwal);
        System.out.println("Jumlah deposito: " + jumlahDeposito);
        System.out.println("Jumlah penarikan: " + jumlahPenarikan);
        System.out.println("Total saldo: Rp " + totalSaldo);
    }
}