package resh11_gmail_com; 

public class HitungSaldoAkhir {
    public static void main(String[] args) {
        double awalSaldo = 200000.0;
        double saldoAkhir;

        double bayarTagihan = -20000.0;
        double depositTunai = 50000.0;

        saldoAkhir = awalSaldo + (bayarTagihan - depositTunai); // Gunakan nilai-nilai di atas

        System.out.println("Saldo Awal: Rp" + awalSaldo);
        System.out.println("Bayar Tagihan: Rp" + bayarTagihan);
        System.out.println("Deposit Tunai: Rp" + depositTunai);
        System.out.println("Saldo Akhir: Rp" + saldoAkhir);
    }
}