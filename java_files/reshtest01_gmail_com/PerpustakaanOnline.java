package reshtest01_gmail_com; 

import java.util.Scanner;

public class PerpustakaanOnline {
    public static void main(String[] args) {
        // 1. Deklarasikan dan inisialisasi variabel
        int bukuDipinjam = 0;
        int bukuBelumPinjam = 0; 
        Scanner input = new Scanner(System.in);

        // 2. Gunakan variable Scanner untuk meminta input
        System.out.println("Masukkan jumlah buku dipinjam: ");
        bukuDipinjam = input.nextInt();
        System.out.println("Masukkan jumlah buku belum pinjam: ");
        bukuBelumPinjam = input.nextInt();

        int totalBuku = 0; 

        // 3. Tampilkan hasil berdasarkan kondisi
        if (bukuDipinjam > 0 && bukuBelumPinjam > 0) {
            System.out.println("Jumlah buku dipinjam: " + bukuDipinjam);
            System.out.println("Jumlah buku belum pinjam: " + bukuBelumPinjam);
            
        } else if (bukuDipinjam == 0 && bukuBelumPinjam > 0) { 
            // UBAH DARI <= MENJADI == (Agar negatif dianggap salah)
            totalBuku = bukuBelumPinjam;
            System.out.println("Tidak ada buku dipinjam, total buku belum pinjam: " + totalBuku);
            
        } else if (bukuDipinjam > 0 && bukuBelumPinjam == 0) { 
            // UBAH DARI <= MENJADI == (Agar negatif dianggap salah)
            totalBuku = bukuDipinjam;
            System.out.println("Tidak ada buku belum dipinjam, total buku pinjam: " + totalBuku);
            
        } else {
            // Kondisi ini akan menangkap:
            // 1. Input (0, 0)
            // 2. Input Negatif (karena tidak masuk logika di atas)
            System.out.println("Mohon maaf tidak ada input yang valid.");
        }
        
        input.close();
    }
}