JAVA_EXAMPLES = [
    {
        "topic": "Tipe Data, Variabel dan Operator",
        "studi_kasus": "Sebuah toko ingin membuat program untuk menghitung total harga barang berdasarkan jumlah item dan harga per item.",
        "tugas": "- Deklarasikan dan inisialisasi variabel hargaPerItem dengan nilai 15000.0\n- Deklarasikan variabel totalHarga\n- Implementasikan perhitungan totalHarga = jumlahItem * hargaPerItem\n- Tampilkan totalHarga pada output",
        "code": """
public class HitungTotalHarga {
    public static void main(String[] args) {
        int jumlahItem = ...;
        
        double hargaPerItem = ...; // Inisialisasi dengan 15000.0
        double totalHarga;

        totalHarga = jumlahItem * ...;

        System.out.println("Detail Pembelian:");
        System.out.println("Jumlah item: " + jumlahItem);
        System.out.println("Harga per item: Rp" + hargaPerItem);
        System.out.println("Total harga: Rp" + totalHarga);
    }
}"""
    },
    {
        "topic": "Tipe Data, Variabel dan Operator",
        "studi_kasus": "Program untuk menghitung luas persegi panjang dengan panjang 10 dan lebar 5.",
        "tugas": "- Deklarasikan dan inisialisasi variabel panjang dengan 10 dan lebar dengan 5\n- Deklarasikan variabel luas\n- Hitung luas = panjang * lebar\n- Tampilkan luas pada output",
        "code": """
public class HitungLuasPersegiPanjang {
    public static void main(String[] args) {
        int panjang = ...;
        int lebar = ...;
        int luas;

        luas = panjang * lebar;

        System.out.println("Panjang: " + panjang);
        System.out.println("Lebar: " + lebar);
        System.out.println("Luas persegi panjang: " + ...);
    }
}"""
    },
    {
        "topic": "Tipe Data, Variabel dan Operator",
        "studi_kasus": "Konversi suhu dari Celsius (25.0) ke Fahrenheit.",
        "tugas": "- Deklarasikan dan inisialisasi suhuCelsius dengan 25.0\n- Deklarasikan variabel suhuFahrenheit\n- Hitung suhuFahrenheit menggunakan rumus: (suhuCelsius * 9/5) + 32\n- Tampilkan suhuCelsius dan suhuFahrenheit",
        "code": """
public class KonversiSuhu {
    public static void main(String[] args) {
        double suhuCelsius = ...; 
        double suhuFahrenheit;

        suhuFahrenheit = (suhuCelsius * 9.0 / ...) + ...;

        System.out.println("Suhu Celsius: " + suhuCelsius + "°C");
        System.out.println("Suhu Fahrenheit: " + suhuFahrenheit + "°F");
    }
}"""
    },
    {
        "topic": "Tipe Data, Variabel dan Operator",
        "studi_kasus": "Menghitung rata-rata dari tiga angka (10, 20, 30).",
        "tugas": "- Deklarasikan dan inisialisasi tiga variabel angka (angka1, angka2, angka3)\n- Hitung total dari ketiga angka\n- Hitung rata-rata\n- Tampilkan rata-rata pada output",
        "code": """
public class HitungRataRata {
    public static void main(String[] args) {
        int angka1 = 10;
        int angka2 = 20;
        int angka3 = ...;

        int total = angka1 + angka2 + angka3;
        double rataRata = total / ...;

        System.out.println("Angka-angka: " + angka1 + ", " + angka2 + ", " + angka3);
        System.out.println("Rata-rata: " + rataRata);
    }
}"""
    },
    {
        "topic": "Tipe Data, Variabel dan Operator",
        "studi_kasus": "Menukar nilai dua variabel tanpa menggunakan variabel sementara.",
        "tugas": "- Deklarasikan dan inisialisasi variabel a dengan 5 dan b dengan 10\n- Tukar nilai a dan b menggunakan operasi aritmatika\n- Tampilkan nilai a dan b setelah pertukaran",
        "code": """
public class TukarNilai {
    public static void main(String[] args) {
        int a = ...;
        int b = ...;

        System.out.println("Sebelum ditukar: a = " + a + ", b = " + b);

        a = a + b;
        b = a - b; 
        a = a - b;

        System.out.println("Setelah ditukar: a = " + a + ", b = " + b);
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 1",
        "studi_kasus": "Sistem penilaian siswa yang menentukan status kelulusan berdasarkan nilai ujian (IF-ELSE).",
        "tugas": "- Deklarasikan variabel status\n- Lengkapi kondisi if dengan syarat nilai >= 70\n- Isi status untuk kondisi else dengan 'TIDAK LULUS'",
        "code": """
public class CekKelulusan {
    public static void main(String[] args) {
        int nilai = ...;
        String ...; // Deklarasikan variabel String

        if (nilai ... 70) { // Lengkapi operator perbandingan (>=)
            status = "..."; // Isi status untuk kondisi LULUS
        } else {
            status = "..."; // Isi status untuk kondisi TIDAK LULUS
        }

        System.out.println("Nilai ujian: " + nilai);
        System.out.println("Status: " + status);
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 1",
        "studi_kasus": "Menentukan apakah sebuah angka adalah bilangan genap atau ganjil.",
        "tugas": "- Deklarasikan variabel angka dengan nilai 7\n- Gunakan operator modulus untuk menentukan genap atau ganjil\n- Tampilkan hasilnya.",
        "code": """
public class CekGenapGanjil {
    public static void main(String[] args) {
        int angka = ...;
        
        if (angka % ... == 0) { // Gunakan operator modulus dengan 2
            System.out.println(angka + " adalah bilangan genap.");
        } else {
            System.out.println(angka + " adalah bilangan ...");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 1",
        "studi_kasus": "Memeriksa apakah seseorang memenuhi syarat untuk memilih berdasarkan usia (minimal 17 tahun).",
        "tugas": "- Deklarasikan dan inisialisasi variabel usia dengan 18\n- Periksa kondisi usia untuk memenuhi syarat memilih\n- Tampilkan pesan yang sesuai.",
        "code": """
public class CekUsiaMemilih {
    public static void main(String[] args) {
        int usia = ...;

        if (usia >= ...) {
            System.out.println("Anda memenuhi syarat untuk memilih.");
        } else {
            System.out.println("Anda belum memenuhi syarat untuk memilih.");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 1",
        "studi_kasus": "Menentukan apakah sebuah tahun adalah tahun kabisat atau bukan.",
        "tugas": "- Deklarasikan dan inisialisasi variabel tahun dengan 2024\n- Gunakan kondisi untuk tahun kabisat (habis dibagi 4, kecuali habis dibagi 100 tapi tidak habis dibagi 400)\n- Tampilkan hasilnya.",
        "code": """
public class CekTahunKabisat {
    public static void main(String[] args) {
        int tahun = ...; // Inisialisasi tahun dengan 2024
        boolean isKabisat = false;

        if ((tahun % 4 == 0 && tahun % 100 != 0) || (tahun % 400 == 0)) { // Kondisi kabisat
            isKabisat = true;
        }

        if (isKabisat) {
            System.out.println(tahun + " adalah tahun kabisat.");
        } else {
            System.out.println(tahun + " bukan tahun kabisat."); // Pesan jika bukan kabisat
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 1",
        "studi_kasus": "Memeriksa apakah sebuah karakter adalah huruf vokal atau konsonan.",
        "tugas": "- Deklarasikan dan inisialisasi variabel karakter dengan 'a'\n- Gunakan kondisi OR untuk memeriksa vokal (a, e, i, o, u)\n- Tampilkan hasilnya.",
        "code": """
public class CekVokalKonsonan {
    public static void main(String[] args) {
        char karakter = '...'; // Inisialisasi dengan 'a'
        
        if (karakter == 'a' || karakter == 'e' || karakter == 'i' || karakter == 'o' || karakter == 'u') { // Lengkapi huruf vokal
            System.out.println(karakter + " adalah huruf vokal.");
        } else {
            System.out.println(karakter + " adalah huruf konsonan."); // Pesan untuk konsonan
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 2",
        "studi_kasus": "Program penilaian huruf berdasarkan nilai numerik siswa dengan skala A, B, C, D, dan E (IF-ELSE IF-ELSE).",
        "tugas": "- Deklarasikan variabel grade\n- Lengkapi kondisi untuk grade B dengan rentang nilai >= 80\n- Isi grade 'C' untuk nilai 70-79\n- Lengkapi kondisi untuk grade D dengan rentang nilai >= 60",
        "code": """
public class PenilaianHuruf {
    public static void main(String[] args) {
        int nilai = ...;
        String grade; // Deklarasikan variabel String

        if (nilai >= 90) {
            grade = "..."; // Isi grade 'A'
        } else if (nilai >= ...) { // Kondisi nilai untuk grade B (80)
            grade = "B";
        } else if (nilai >= 70) {
            grade = "..."; // Isi nilai grade 'C'
        } else if (nilai >= ...) { // Kondisi nilai untuk grade D (60)
            grade = "D";
        } else {
            grade = "..."; // Isi grade 'E'
        }

        System.out.println("Nilai: " + nilai);
        System.out.println("Grade: " + grade);
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 2",
        "studi_kasus": "Program kalkulator sederhana yang melakukan operasi aritmatika berdasarkan pilihan operator (switch-case).",
        "tugas": "- Deklarasikan variabel hasil\n- Inisialisasi operator dengan karakter '+'\n- Lengkapi case untuk pengurangan dengan karakter '-'\n- Lengkapi case untuk pembagian dengan karakter '/'",
        "code": """
public class KalkulatorSederhana {
    public static void main(String[] args) {
        double bilangan1 = 20.0;
        double bilangan2 = 4.0;
        char operator = '...'; // Inisialisasi dengan karakter '+'
        double hasil; // Deklarasikan variabel hasil

        switch(operator) {
            case '+':
                hasil = bilangan1 + ...;
                break;
            case '...': // Karakter untuk pengurangan (-)
                hasil = bilangan1 - bilangan2;
                break;
            case '*':
                hasil = bilangan1 * bilangan2;
                break;
            case '...': // Karakter untuk pembagian (/)
                hasil = bilangan1 / bilangan2;
                break;
            default:
                System.out.println("Operator tidak valid");
                return;
        }

        System.out.println(bilangan1 + " " + operator + " " + bilangan2 + " = " + hasil);
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 2",
        "studi_kasus": "Menentukan hari dalam seminggu berdasarkan angka (1=Senin, 7=Minggu) menggunakan switch-case.",
        "tugas": "- Deklarasikan dan inisialisasi variabel hariAngka dengan 3\n- Deklarasikan variabel namaHari\n- Lengkapi setiap case dengan nama hari yang sesuai\n- Tampilkan nama hari.",
        "code": """
public class TentukanHari {
    public static void main(String[] args) {
        int hariAngka = ...; // Inisialisasi hariAngka dengan 3
        String namaHari;

        switch (hariAngka) {
            case 1:
                namaHari = "..."; // Isi nama hari "Senin"
                break;
            case 2:
                namaHari = "Selasa";
                break;
            case 3:
                namaHari = "..."; // Isi nama hari "Rabu"
                break;
            case 4:
                namaHari = "Kamis";
                break;
            case 5:
                namaHari = "..."; // Isi nama hari "Jumat"
                break;
            case 6:
                namaHari = "Sabtu";
                break;
            case 7:
                namaHari = "..."; // Isi nama hari "Minggu"
                break;
            default:
                namaHari = "..."; // Isi pesan default "Tidak Valid"
                break;
        }
        System.out.println("Hari ke-" + hariAngka + " adalah " + namaHari);
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 2",
        "studi_kasus": "Menentukan musim berdasarkan bulan (1-12) dan belahan bumi ('U'/'S') menggunakan if-else if-else dan operator logika.",
        "tugas": "- Deklarasikan dan inisialisasi variabel bulan (1-12) dan belahanBumi ('U'/'S')\n- Gunakan kombinasi if-else if-else dan operator logika (&&, ||) untuk menentukan musim\n- Tampilkan musim yang sesuai.",
        "code": """
public class TentukanMusim {
    public static void main(String[] args) {
        int bulan = 6;
        char belahanBumi = 'U'; // 'U' untuk Utara, 'S' untuk Selatan
        String musim;

        if (belahanBumi == 'U') {
            if (bulan >= 3 && bulan <= 5) {
                musim = "Musim Semi";
            } else if (bulan >= 6 && bulan <= 8) { // Bulan akhir musim panas (8)
                musim = "Musim Panas";
            } else if (bulan >= 9 && bulan <= 11) {
                musim = "Musim Gugur";
            } else {
                musim = "Musim Dingin";
            }
        } else if (belahanBumi == 'S') { // Kondisi untuk belahan bumi selatan
            if (bulan >= 3 && bulan <= 5) {
                musim = "Musim Gugur";
            } else if (bulan >= 6 && bulan <= 8) {
                musim = "Musim Dingin";
            } else if (bulan >= 9 && bulan <= 11) {
                musim = "Musim Semi";
            } else {
                musim = "Musim Panas"; // Musim panas untuk belahan selatan (Des-Feb)
            }
        } else {
            musim = "Belahan bumi tidak valid.";
        }
        System.out.println("Pada bulan " + bulan + " di belahan bumi " + belahanBumi + " adalah musim: " + musim);
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 2",
        "studi_kasus": "Menentukan jenis segitiga (sama sisi, sama kaki, atau sembarang) berdasarkan panjang sisi-sisinya.",
        "tugas": "- Deklarasikan dan inisialisasi variabel sisi1, sisi2, sisi3\n- Gunakan if-else if-else untuk menentukan jenis segitiga\n- Tampilkan jenis segitiga.",
        "code": """
public class JenisSegitiga {
    public static void main(String[] args) {
        int sisi1 = 5;
        int sisi2 = 5;
        int sisi3 = ..; // Inisialisasi dengan 7

        if (sisi1 == sisi2 && sisi2 == sisi3) {
            System.out.println("Ini adalah segitiga sama sisi.");
        } else if (sisi1 == sisi2 || sisi1 == sisi3 || sisi2 == sisi3) { // Kondisi sisi sama kaki
            System.out.println("Ini adalah segitiga sama kaki.");
        } else {
            System.out.println("Ini adalah segitiga sembarang."); // Pesan untuk segitiga sembarang
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan Bersarang",
        "studi_kasus": "Sistem penilaian dengan kategori berdasarkan nilai dan kehadiran siswa (IF bersarang).",
        "tugas": "- Deklarasikan variabel nilai dengan 85 dan kehadiran dengan 90\n- Gunakan if bersarang untuk memeriksa nilai >= 70 terlebih dahulu\n- Di dalam if pertama, periksa kehadiran >= 80 untuk kategori 'LULUS DENGAN BAIK'\n- Lengkapi kondisi else untuk kehadiran < 80 dengan 'LULUS'\n- Isi kondisi else utama dengan 'TIDAK LULUS'",
        "code": """
public class PenilaianBersarang {
    public static void main(String[] args) {
        int nilai = ...; // Inisialisasi nilai dengan 85
        int kehadiran = ...; // Inisialisasi kehadiran dengan 90
        String kategori;

        if (nilai >= ...) { // Kondisi utama untuk nilai >= 70
            if (kehadiran >= ...) { // Kondisi bersarang untuk kehadiran >= 80
                kategori = "..."; // Isi kategori "LULUS DENGAN BAIK"
            } else {
                kategori = "..."; // Kondisi jika kehadiran < 80 ("LULUS")
            }
        } else {
            kategori = "..."; // Kondisi jika nilai < 70 ("TIDAK LULUS")
        }

        System.out.println("Nilai: " + nilai);
        System.out.println("Kehadiran: " + kehadiran + "%");
        System.out.println("Kategori: " + kategori);
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan Bersarang",
        "studi_kasus": "Program untuk menentukan diskon berdasarkan jenis membership dan total belanja.",
        "tugas": "- Deklarasikan variabel totalBelanja dengan 150000 dan jenisMember dengan 'GOLD'\n- Gunakan if bersarang untuk memeriksa jenis membership terlebih dahulu\n- Di dalam setiap kondisi membership, periksa total belanja untuk menentukan diskon\n- Hitung harga setelah diskon dan tampilkan hasilnya.",
        "code": """
public class DiskonBersarang {
    public static void main(String[] args) {
        double totalBelanja = ...; // Total belanja (150000)
        String jenisMember = "..."; // Jenis membership ("GOLD")
        double diskonPersen = 0;

        if (jenisMember.equals("PLATINUM")) {
            if (totalBelanja >= 200000) {
                diskonPersen = ...; // Diskon 20% untuk PLATINUM dengan belanja >= 200k
            } else {
                diskonPersen = ...; // Diskon 15% untuk PLATINUM dengan belanja < 200k
            }
        } else if (jenisMember.equals("...")) { // Kondisi untuk GOLD member
            if (totalBelanja >= 100000) {
                diskonPersen = ...; // Diskon 10% untuk GOLD dengan belanja >= 100k
            } else {
                diskonPersen = ...; // Diskon 5% untuk GOLD dengan belanja < 100k
            }
        } else {
            if (totalBelanja >= 50000) {
                diskonPersen = ...; // Diskon 2% untuk member biasa dengan belanja >= 50k
            }
        }

        double jumlahDiskon = totalBelanja * diskonPersen / 100;
        double hargaAkhir = totalBelanja - jumlahDiskon;

        System.out.println("=== PERHITUNGAN DISKON ===");
        System.out.println("Jenis Member: " + jenisMember);
        System.out.println("Total Belanja: Rp" + totalBelanja);
        System.out.println("Diskon: " + diskonPersen + "% (Rp" + jumlahDiskon + ")");
        System.out.println("Harga Akhir: Rp" + hargaAkhir);
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan Bersarang",
        "studi_kasus": "Menentukan kategori BMI dengan klasifikasi berdasarkan usia dan berat badan ideal.",
        "tugas": "- Deklarasikan variabel berat, tinggi, dan usia\n- Hitung BMI menggunakan rumus berat / (tinggi * tinggi)\n- Gunakan if bersarang untuk mengkategorikan BMI berdasarkan usia\n- Tampilkan kategori BMI yang sesuai.",
        "code": """
public class KategoriBMI {
    public static void main(String[] args) {
        double berat = ...; // Berat badan dalam kg (70.0)
        double tinggi = ...; // Tinggi badan dalam meter (1.75)
        int usia = ...; // Usia dalam tahun (25)
        
        double bmi = berat / (tinggi * tinggi); // Hitung BMI
        String kategori;

        if (usia >= ...) { // Kategori untuk dewasa (>= 18)
            if (bmi < 18.5) {
                kategori = "Kurus";
            } else if (bmi < ...) { // Kondisi untuk BMI normal (< 25.0)
                kategori = "Normal";
            } else if (bmi < 30.0) {
                kategori = "Gemuk";
            } else {
                kategori = "Obesitas";
            }
        } else { // Kategori untuk remaja (< 18 tahun)
            if (bmi < 16.0) {
                kategori = "Kurus (Remaja)";
            } else if (bmi < ...) { // Kategori normal untuk remaja (< 23.0)
                kategori = "..."; // Kategori normal untuk remaja
            } else {
                kategori = "Gemuk (Remaja)";
            }
        }

        System.out.println("=== ANALISIS BMI ===");
        System.out.println("Usia: " + usia + " tahun");
        System.out.println("Berat: " + berat + " kg");
        System.out.println("Tinggi: " + tinggi + " m");
        System.out.println("BMI: " + String.format("%.2f", bmi));
        System.out.println("Kategori: " + kategori);
    }
}"""
    },
    {
        "topic": "Perulangan While",
        "studi_kasus": "Program untuk menghitung jumlah digit dari sebuah bilangan menggunakan while loop.",
        "tugas": "- Deklarasikan variabel bilangan dengan 12345\n- Inisialisasi jumlahDigit dengan 0 dan temp dengan nilai bilangan\n- Gunakan while loop dengan kondisi temp > 0\n- Di dalam loop, increment jumlahDigit dan bagi temp dengan 10\n- Tampilkan hasil jumlah digit.",
        "code": """
public class HitungDigitWhile {
    public static void main(String[] args) {
        int bilangan = 12345; // Bilangan yang akan dihitung digitnya
        int jumlahDigit = 0; // Inisialisasi counter digit
        int temp = bilangan; // Variabel temporary

        System.out.println("Menghitung jumlah digit dari: " + bilangan);
        
        while (temp > 0) { // Kondisi perulangan
            jumlahDigit++; // Increment counter
            temp = temp / 10; // Hilangkan digit terakhir
            System.out.println("Sisa: " + temp + ", Jumlah digit sejauh ini: " + jumlahDigit);
        }

        System.out.println("Total jumlah digit: " + jumlahDigit);
    }
}"""
    },
    {
        "topic": "Perulangan While",
        "studi_kasus": "Program validasi input password sederhana menggunakan while loop.",
        "tugas": "- Deklarasikan variabel passwordBenar dengan '12345'\n- Inisialisasi inputPassword dengan nilai yang salah untuk simulasi\n- Gunakan while loop untuk memvalidasi password\n- Simulasikan input yang benar setelah beberapa percobaan\n- Tampilkan pesan sukses ketika password benar.",
        "code": """
public class ValidasiPassword {
    public static void main(String[] args) {
        String passwordBenar = "12345"; // Password yang benar
        String inputPassword = "wrong"; // Simulasi input awal yang salah
        int percobaan = 0;

        System.out.println("=== VALIDASI PASSWORD ===");
        
        while (!inputPassword.equals(passwordBenar)) { // Kondisi perulangan
            percobaan++; // Increment percobaan
            System.out.println("Percobaan ke-" + percobaan + ": Password salah!");
            
            // Simulasi input baru (dalam aplikasi nyata menggunakan Scanner)
            if (percobaan == 1) {
                inputPassword = "54321"; // Percobaan kedua masih salah
            } else if (percobaan == 2) {
                inputPassword = "12345"; // Percobaan ketiga benar
            }
        }
        
        System.out.println("Password benar! Akses diberikan setelah " + percobaan + " percobaan.");
    }
}"""
    },
    {
        "topic": "Perulangan While",
        "studi_kasus": "Menampilkan deret Fibonacci menggunakan while loop hingga nilai tertentu.",
        "tugas": "- Inisialisasi variabel untuk deret Fibonacci (a=0, b=1)\n- Set batas maksimal nilai Fibonacci (contoh: 100)\n- Gunakan while loop untuk generate deret selama nilai <= batas\n- Tampilkan setiap nilai dalam deret\n- Hitung total berapa angka yang ditampilkan.",
        "code": """
public class FibonacciWhile {
    public static void main(String[] args) {
        int a = 0; // Nilai Fibonacci pertama
        int b = 1; // Nilai Fibonacci kedua
        int batas = 100; // Batas maksimal nilai
        int count = 0; // Counter jumlah angka

        System.out.println("Deret Fibonacci hingga " + batas + ":");
        System.out.print(a + " "); // Tampilkan nilai pertama
        count++;

        while (b <= batas) { // Kondisi perulangan
            System.out.print(b + " ");
            count++; // Increment counter
            
            int temp = a + b; // Hitung nilai Fibonacci berikutnya
            a = b; // Update nilai a
            b = temp; // Update nilai b
        }
        
        System.out.println("\\nTotal angka dalam deret: " + count);
    }
}"""
    },
    {
        "topic": "Perulangan Do-While",
        "studi_kasus": "Program menu interaktif yang terus berjalan hingga user memilih keluar.",
        "tugas": "- Deklarasikan variabel pilihan\n- Gunakan do-while loop untuk menampilkan menu\n- Simulasikan pilihan user dengan mengubah nilai pilihan di setiap iterasi\n- Tambahkan switch-case untuk menangani setiap pilihan menu\n- Loop berhenti ketika pilihan = 0",
        "code": """
public class MenuInteraktif {
    public static void main(String[] args) {
        int pilihan; // Variabel untuk menyimpan pilihan user
        int iterasi = 0; // Counter untuk simulasi

        do {
            System.out.println("\\n=== MENU UTAMA ===");
            System.out.println("1. Lihat Profil");
            System.out.println("2. Edit Data");
            System.out.println("3. Laporan");
            System.out.println("0. Keluar");
            System.out.print("Pilihan Anda: ");
            
            // Simulasi input user (dalam aplikasi nyata menggunakan Scanner)
            iterasi++;
            if (iterasi == 1) {
                pilihan = 1; // Simulasi pilih menu 1
            } else if (iterasi == 2) {
                pilihan = 2; // Simulasi pilih menu 2
            } else if (iterasi == 3) {
                pilihan = 3; // Simulasi pilih menu 3
            } else {
                pilihan = 0; // Simulasi pilih keluar
            }
            
            System.out.println(pilihan); // Tampilkan pilihan
            
            switch (pilihan) { // Switch-case untuk menangani pilihan
                case 1:
                    System.out.println("→ Menampilkan profil pengguna...");
                    break;
                case 2:
                    System.out.println("→ Membuka form edit data...");
                    break;
                case 3:
                    System.out.println("→ Generating laporan...");
                    break;
                case 0:
                    System.out.println("→ Terima kasih! Program selesai.");
                    break;
                default:
                    System.out.println("→ Pilihan tidak valid!");
            }
            
        } while (pilihan != 0); // Kondisi perulangan
    }
}"""
    },
    {
        "topic": "Perulangan Do-While",
        "studi_kasus": "Program tebak angka sederhana dengan do-while loop.",
        "tugas": "- Set angka rahasia yang harus ditebak (contoh: 7)\n- Gunakan do-while untuk meminta tebakan user\n- Simulasikan tebakan user dengan array nilai\n- Berikan feedback apakah tebakan terlalu tinggi, rendah, atau benar\n- Loop berlanjut hingga tebakan benar.",
        "code": """
public class TebakAngka {
    public static void main(String[] args) {
        int angkaRahasia = 7; // Angka yang harus ditebak
        int tebakan; // Variabel untuk tebakan user
        int percobaan = 0; // Counter percobaan
        
        // Array untuk simulasi tebakan user
        int[] tebakanSimulasi = {5, 9, 3, 7}; // Simulasi beberapa tebakan
        
        System.out.println("=== GAME TEBAK ANGKA (1-10) ===");
        System.out.println("Tebak angka antara 1 sampai 10!");
        
        do {
            System.out.print("\\nTebakan Anda: ");
            
            // Simulasi input (dalam aplikasi nyata menggunakan Scanner)
            if (percobaan < tebakanSimulasi.length) {
                tebakan = tebakanSimulasi[percobaan]; // Ambil tebakan dari array
            } else {
                tebakan = angkaRahasia; // Fallback jika array habis
            }
            
            System.out.println(tebakan);
            percobaan++; // Increment percobaan
            
            if (tebakan < angkaRahasia) {
                System.out.println("Terlalu rendah! Coba angka yang lebih besar.");
            } else if (tebakan > angkaRahasia) { // Kondisi tebakan terlalu tinggi
                System.out.println("Terlalu tinggi! Coba angka yang lebih kecil.");
            } else {
                System.out.println("🎉 BENAR! Anda menebak dengan benar!");
                System.out.println("Jumlah percobaan: " + percobaan);
            }
            
        } while (tebakan != angkaRahasia); // Kondisi perulangan
    }
}"""
    },
    {
        "topic": "Perulangan Do-While",
        "studi_kasus": "Validasi input angka positif menggunakan do-while loop.",
        "tugas": "- Deklarasikan variabel untuk menyimpan input angka\n- Gunakan do-while untuk meminta input hingga valid\n- Simulasikan input dengan array yang berisi nilai negatif dan positif\n- Validasi bahwa input harus berupa angka positif (> 0)\n- Tampilkan pesan error untuk input yang tidak valid.",
        "code": """
public class ValidasiAngkaPositif {
    public static void main(String[] args) {
        int angka; // Variabel untuk input angka
        int indeksSimulasi = 0; // Index untuk simulasi input
        
        // Array simulasi input (negatif, nol, positif)
        int[] inputSimulasi = {-5, 0, -10, 25}; // Simulasi berbagai input
        
        System.out.println("=== VALIDASI ANGKA POSITIF ===");
        System.out.println("Masukkan angka positif (> 0):");
        
        do {
            System.out.print("Input: ");
            
            // Simulasi input user
            if (indeksSimulasi < inputSimulasi.length) {
                angka = inputSimulasi[indeksSimulasi]; // Ambil dari array simulasi
                indeksSimulasi++;
            } else {
                angka = 1; // Fallback ke angka valid
            }
            
            System.out.println(angka);
            
            if (angka <= 0) { // Validasi angka positif
                System.out.println("❌ Error: Angka harus positif (> 0)! Coba lagi.");
            } else {
                System.out.println("✅ Input valid! Angka " + angka + " diterima.");
            }
            
        } while (angka <= 0); // Kondisi perulangan
        
        System.out.println("\\nProses validasi selesai. Angka final: " + angka);
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 1",
        "studi_kasus": "Program untuk mencetak tabel perkalian dari angka yang ditentukan (for loop).",
        "tugas": "- Set batas perulangan i <= 10\n- Tampilkan variabel i pada output perkalian",
        "code": """
public class TabelPerkalian {
    public static void main(String[] args) {
        int angka = 7;
        int batasPerkalian = 10;

        System.out.println("Tabel perkalian " + angka + ":");

        for(int i = 1; i <= ...; i++) { // Batas akhir perulangan (10)
            int hasil = angka * i;
            System.out.println(angka + " x " + ... + " = " + hasil); // Tampilkan variabel 'i'
        }
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 1",
        "studi_kasus": "Program untuk menghitung jumlah digit dari sebuah bilangan (while loop).",
        "tugas": "- Inisialisasi jumlahDigit dengan nilai 0\n- Inisialisasi temp dengan nilai bilangan\n- Lengkapi kondisi while dengan temp > 0",
        "code": """
public class HitungJumlahDigit {
    public static void main(String[] args) {
        int bilangan = 12345;
        int jumlahDigit = 0; // Inisialisasi dengan 0
        int temp = bilangan; // Inisialisasi dengan nilai bilangan

        while(...) { // Kondisi perulangan (temp > 0)
            jumlahDigit++;
            temp = temp / 10;
        }

        System.out.println("Bilangan: " + bilangan);
        System.out.println("Jumlah digit: " + jumlahDigit);
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 1",
        "studi_kasus": "Mencetak angka genap dari 1 hingga 20 menggunakan for loop.",
        "tugas": "- Gunakan for loop untuk iterasi dari 1 hingga 20\n- Gunakan kondisi if dan operator modulus untuk memeriksa bilangan genap\n- Cetak bilangan genap.",
        "code": """
public class CetakAngkaGenap {
    public static void main(String[] args) {
        System.out.println("Angka genap dari 1 sampai 20:");
        for (int i = 1; i <= ...; i++) { // Batas atas perulangan (20)
            if (i % ... == 0) { // Kondisi untuk bilangan genap (modulus 2)
                System.out.println(i);
            }
        }
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 1",
        "studi_kasus": "Menghitung jumlah total dari angka 1 sampai 100 menggunakan for loop.",
        "tugas": "- Deklarasikan dan inisialisasi variabel totalJumlah dengan 0\n- Gunakan for loop untuk menjumlahkan angka dari 1 hingga 100\n- Tampilkan totalJumlah.",
        "code": """
public class JumlahDeretAngka {
    public static void main(String[] args) {
        int totalJumlah = ...; // Inisialisasi totalJumlah dengan 0

        for (int i = 1; i <= ...; i++) { // Batas atas perulangan (100)
            totalJumlah += i;
        }
        System.out.println("Jumlah angka dari 1 sampai 100 adalah: " + ...); // Tampilkan total (totalJumlah)
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 1",
        "studi_kasus": "Mencetak angka dari 10 sampai 1 secara menurun menggunakan while loop.",
        "tugas": "- Deklarasikan dan inisialisasi variabel hitungMundur dengan 10\n- Gunakan while loop untuk mencetak angka selama hitungMundur lebih dari 0\n- Kurangi nilai hitungMundur di setiap iterasi.",
        "code": """
public class HitungMundur {
    public static void main(String[] args) {
        int hitungMundur = ...; // Inisialisasi dengan 10

        while (hitungMundur > ...) { // Kondisi perulangan (> 0)
            System.out.println(hitungMundur);
            hitungMundur--; // Kurangi nilai
        }
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 2",
        "studi_kasus": "Program menu sederhana yang terus berjalan sampai user memilih keluar (do-while loop).",
        "tugas": "- Deklarasikan variabel pilihan\n- Set pilihan dengan nilai 1 untuk simulasi (atau gunakan Scanner untuk input user)\n- Lengkapi kondisi while dengan pilihan != 0",
        "code": """
import java.util.Scanner;

public class MenuSederhana {
    public static void main(String[] args) {
        int pilihan; // Deklarasikan variabel
        // Scanner scanner = new Scanner(System.in); // Opsional: Untuk input user

        do {
            System.out.println("=== MENU UTAMA ===");
            System.out.println("1. Pilihan 1");
            System.out.println("2. Pilihan 2");
            System.out.println("3. Pilihan 3");
            System.out.println("0. Keluar");
            System.out.print("Masukkan pilihan: ");
            
            pilihan = 1; // Isi dengan angka (contoh: 1) atau scanner.nextInt()
            
            switch(pilihan) {
                case 1:
                    System.out.println("Anda memilih pilihan 1");
                    break;
                case 2:
                    System.out.println("Anda memilih pilihan 2");
                    break;
                case 3:
                    System.out.println("Anda memilih pilihan 3");
                    break;
                case 0:
                    System.out.println("Terima kasih!");
                    break;
                default:
                    System.out.println("Pilihan tidak valid");
            }
            // Tambahan untuk demo agar loop berhenti:
            if (pilihan == 1) pilihan = 0; 
        } while(pilihan != 0); // Kondisi perulangan (variabel 'pilihan' tidak sama dengan 0)
        // scanner.close(); // Opsional: Tutup scanner jika digunakan
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 2",
        "studi_kasus": "Program untuk mencetak pola piramida angka dengan tinggi yang dapat disesuaikan (nested loops).",
        "tugas": "- Inisialisasi tinggi piramida dengan nilai 5\n- Lengkapi batas loop spasi dengan (tinggi - i)\n- Set batas loop angka dengan nilai i\n- Cetak nilai i pada setiap posisi",
        "code": """
public class PiramidaAngka {
    public static void main(String[] args) {
        int tinggi = 5; // Inisialisasi tinggi dengan 5

        System.out.println("Pola piramida angka:");

        for(int i = 1; i <= tinggi; i++) {
            
            for(int j = 1; j <= (tinggi - i); j++) { // Gunakan variabel 'tinggi' dan 'i'
                System.out.print(" ");
            }
            
            for(int k = 1; k <= i; k++) { // Batas loop angka (variabel 'i')
                System.out.print(i + " "); // Bantuan 4: Cetak nilai 'i'
            }
            
            System.out.println();
        }
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 2",
        "studi_kasus": "Menampilkan pola bintang persegi dengan ukuran 4x4 menggunakan perulangan bersarang.",
        "tugas": "- Gunakan perulangan for bersarang\n- Cetak karakter '*' untuk setiap posisi\n- Pindahkan ke baris baru setelah setiap baris selesai.",
        "code": """
public class PolaPersegiBintang {
    public static void main(String[] args) {
        int ukuran = 4;
        System.out.println("Pola persegi bintang:");

        for (int i = 1; i <= ukuran; i++) { // Loop untuk baris
            for (int j = 1; j <= ukuran; j++) { // Loop untuk kolom
                System.out.print("* "); // Cetak karakter '*'
            }
            System.out.println(); // Pindah ke baris baru
        }
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 2",
        "studi_kasus": "Mencetak pola segitiga siku-siku bintang dengan tinggi 5.",
        "tugas": "- Gunakan perulangan for bersarang\n- Loop luar untuk baris, loop dalam untuk mencetak bintang\n- Sesuaikan jumlah bintang per baris",
        "code": """
public class PolaSegitigaBintang {
    public static void main(String[] args) {
        int tinggi = 5;
        System.out.println("Pola segitiga siku-siku bintang:");

        for (int i = 1; i <= tinggi; i++) { // Loop untuk baris
            for (int j = 1; j <= i; j++) { // Loop untuk jumlah bintang
                System.out.print("*");
            }
            System.out.println(); // Pindah ke baris baru
        }
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 2",
        "studi_kasus": "Menampilkan bilangan prima antara 1 dan 20 menggunakan perulangan bersarang.",
        "tugas": "- Gunakan loop luar untuk angka 1 sampai 20\n- Gunakan loop dalam untuk memeriksa apakah angka adalah prima\n- Tampilkan angka yang prima.",
        "code": """
public class BilanganPrima {
    public static void main(String[] args) {
        System.out.println("Bilangan prima antara 1 dan 20:");
        for (int i = 2; i <= 20; i++) { // Angka yang akan diperiksa
            boolean isPrima = true;
            for (int j = 2; j <= i / 2; j++) { // Batas loop untuk cek prima
                if (i % j == 0) {
                    isPrima = false;
                    break;
                }
            }
            if (isPrima) { // Jika bilangan prima
                System.out.print(i + " ");
            }
        }
        System.out.println();
    }
}"""
    },
    {
        "topic": "Array 1",
        "studi_kasus": "Program untuk menghitung rata-rata nilai mahasiswa dan menentukan nilai tertinggi dan terendah (array satu dimensi).",
        "tugas": "- Deklarasikan dan inisialisasi variabel total dengan 0\n- Inisialisasi nilaiTerendah dengan nilai[0]\n- Lengkapi kondisi untuk mencari nilai terendah dengan nilai[i] < nilaiTerendah\n- Hitung rata-rata dengan membagi total dengan nilai.length",
        "code": """
public class StatistikNilai {
    public static void main(String[] args) {
        int[] nilai = {85, 92, 78, 96, 88};

        int total = ...; // Inisialisasi total (0)
        int nilaiTertinggi = nilai[0];
        int nilaiTerendah = nilai[...]; // Inisialisasi dengan nilai[0]

        for(int i = 0; i < nilai.length; i++) {
            total += nilai[i];
            
            if(nilai[i] > nilaiTertinggi) {
                nilaiTertinggi = nilai[i];
            }
            
            if(nilai[i] ... nilaiTerendah) { // Lengkapi kondisi (<)
                nilaiTerendah = nilai[i];
            }
        }

        double rataRata = (double) total / ...; // Bagi dengan panjang array (nilai.length)

        System.out.println("=== STATISTIK NILAI MAHASISWA ===");
        System.out.print("Nilai: ");
        for(int n : nilai) {
            System.out.print(n + " ");
        }
        System.out.println();
        System.out.println("Total: " + total);
        System.out.println("Rata-rata: " + rataRata);
        System.out.println("Nilai tertinggi: " + nilaiTertinggi);
        System.out.println("Nilai terendah: " + nilaiTerendah);
    }
}"""
    },
    {
        "topic": "Array 1",
        "studi_kasus": "Mencari elemen tertentu dalam array dan memberitahu posisinya.",
        "tugas": "- Deklarasikan dan inisialisasi array angka\n- Deklarasikan dan inisialisasi elemen yang dicari\n- Gunakan loop untuk mencari elemen\n- Tampilkan pesan jika ditemukan atau tidak ditemukan.",
        "code": """
public class CariElemenArray {
    public static void main(String[] args) {
        int[] angka = {10, 20, 30, 40, 50}; // Inisialisasi array
        int elemenCari = ...; // Elemen yang dicari (30)
        boolean ditemukan = false;
        int posisi = -1;

        for (int i = 0; i < angka.length; i++) {
            if (angka[i] == ...) { // Bandingkan elemen array dengan elemenCari
                ditemukan = true;
                posisi = i;
                break;
            }
        }

        if (...) { // Kondisi jika ditemukan (ditemukan)
            System.out.println("Elemen " + elemenCari + " ditemukan pada posisi indeks " + posisi);
        } else {
            System.out.println("Elemen " + elemenCari + " tidak ditemukan dalam array.");
        }
    }
}"""
    },
    {
        "topic": "Array 1",
        "studi_kasus": "Menjumlahkan semua elemen dalam array satu dimensi.",
        "tugas": "- Deklarasikan dan inisialisasi array data\n- Deklarasikan dan inisialisasi variabel jumlah dengan 0\n- Gunakan loop untuk menjumlahkan setiap elemen\n- Tampilkan total jumlah.",
        "code": """
public class JumlahElemenArray {
    public static void main(String[] args) {
        int[] data = {1, 2, 3, 4, 5}; // Inisialisasi array
        int jumlah = ...; // Inisialisasi jumlah dengan 0

        for (int i = 0; i < data.length; i++) {
            jumlah += data[...]; // Tambahkan setiap elemen array ke jumlah (data[i])
        }

        System.out.println("Elemen array: ");
        for (int n : data) {
            System.out.print(n + " ");
        }
        System.out.println("\\nTotal jumlah elemen: " + jumlah);
    }
}"""
    },
    {
        "topic": "Array 1",
        "studi_kasus": "Membalik urutan elemen dalam sebuah array.",
        "tugas": "- Deklarasikan dan inisialisasi array original\n- Buat array baru untuk menyimpan hasil balik\n- Salin elemen dari original ke array baru secara terbalik\n- Tampilkan kedua array.",
        "code": """
public class BalikArray {
    public static void main(String[] args) {
        int[] original = {1, 2, 3, 4, 5};
        int[] reversed = new int[original.length];

        System.out.print("Array asli: ");
        for (int i : original) System.out.print(i + " ");
        System.out.println();

        for (int i = 0; i < original.length; i++) {
            reversed[i] = original[original.length - 1 - i]; // Balik indeks
        }

        System.out.print("Array terbalik: ");
        for (int i : reversed) System.out.print(i + " "); // Tampilkan array terbalik
        System.out.println();
    }
}"""
    },
    {
        "topic": "Array 1",
        "studi_kasus": "Menyalin elemen dari satu array ke array lain.",
        "tugas": "- Deklarasikan dan inisialisasi array sumber\n- Deklarasikan array tujuan dengan ukuran yang sama\n- Gunakan loop untuk menyalin elemen\n- Tampilkan kedua array.",
        "code": """
public class SalinArray {
    public static void main(String[] args) {
        int[] sumber = {10, 20, 30};
        int[] tujuan = new int[sumber.length]; // Array tujuan

        System.out.print("Array sumber: ");
        for (int i : sumber) System.out.print(i + " ");
        System.out.println();

        for (int i = 0; i < sumber.length; i++) {
            tujuan[i] = sumber[i]; // Salin elemen
        }

        System.out.print("Array tujuan: ");
        for (int i : tujuan) System.out.print(i + " "); // Tampilkan array tujuan
        System.out.println();
    }
}"""
    },
    {
        "topic": "Array Multidimensi",
        "studi_kasus": "Program untuk menghitung jumlah elemen dalam matriks 2D dan mencari elemen terbesar.",
        "tugas": "- Inisialisasi variabel jumlahTotal dengan 0\n- Lengkapi matriks dengan nilai 5 dan 9\n- Lengkapi kondisi untuk mencari nilai terbesar dengan matriks[i][j] > nilaiTerbesar",
        "code": """
public class OperasiMatriks {
    public static void main(String[] args) {
        int[][] matriks = {
            {1, 2, 3},
            {4, 5, 6}, // Isi dengan 5
            {7, 8, 9}  // Isi dengan 9
        };

        int jumlahTotal = 0; // Inisialisasi variabel
        int nilaiTerbesar = matriks[0][0];

        System.out.println("Matriks 3x3:");

        for(int i = 0; i < matriks.length; i++) {
            for(int j = 0; j < matriks[i].length; j++) {
                System.out.print(matriks[i][j] + " ");
                jumlahTotal += matriks[i][j];
                
                if(matriks[i][j] > nilaiTerbesar) { // Bantuan 4: Lengkapi kondisi
                    nilaiTerbesar = matriks[i][j];
                }
            }
            System.out.println();
        }

        System.out.println("Jumlah semua elemen: " + jumlahTotal);
        System.out.println("Nilai terbesar: " + nilaiTerbesar);
    }
}"""
    },
    {
        "topic": "Array Multidimensi",
        "studi_kasus": "Menjumlahkan dua matriks 2x2.",
        "tugas": "- Deklarasikan dan inisialisasi dua matriks 2x2 (matriksA dan matriksB)\n- Deklarasikan matriks hasil dengan ukuran yang sama\n- Lakukan penjumlahan elemen per elemen\n- Tampilkan matriks hasil.",
        "code": """
public class PenjumlahanMatriks {
    public static void main(String[] args) {
        int[][] matriksA = {{1, 2}, {3, 4}};
        int[][] matriksB = {{5, 6}, {7, 8}};
        int[][] matriksHasil = new int[2][2]; // Ukuran kolom

        System.out.println("Matriks A:");
        for (int i = 0; i < matriksA.length; i++) {
            for (int j = 0; j < matriksA[i].length; j++) {
                System.out.print(matriksA[i][j] + " ");
            }
            System.out.println();
        }

        System.out.println("\\nMatriks B:");
        for (int i = 0; i < matriksB.length; i++) {
            for (int j = 0; j < matriksB[i].length; j++) {
                System.out.print(matriksB[i][j] + " ");
            }
            System.out.println();
        }

        System.out.println("\\nHasil Penjumlahan Matriks:");
        for (int i = 0; i < matriksA.length; i++) {
            for (int j = 0; j < matriksA[i].length; j++) {
                matriksHasil[i][j] = matriksA[i][j] + matriksB[i][j]; // Jumlahkan elemen dari matriksB
                System.out.print(matriksHasil[i][j] + " ");
            }
            System.out.println();
        }
    }
}"""
    },
    {
        "topic": "Array Multidimensi",
        "studi_kasus": "Mengakses dan menampilkan elemen pada baris dan kolom tertentu dalam matriks 3x3.",
        "tugas": "- Deklarasikan dan inisialisasi matriks 3x3\n- Set nilai untuk barisTarget dan kolomTarget\n- Akses dan tampilkan elemen pada posisi tersebut.",
        "code": """
public class AksesElemenMatriks {
    public static void main(String[] args) {
        int[][] matriks = {
            {10, 20, 30},
            {40, 50, 60},
            {70, 80, 90}
        };

        int barisTarget = ...; // Baris yang ingin diakses (indeks 0-2, contoh: 1)
        int kolomTarget = ...; // Kolom yang ingin diakses (indeks 0-2, contoh: 2)

        // Pastikan indeks valid
        if (barisTarget >= 0 && barisTarget < matriks.length && 
            kolomTarget >= 0 && kolomTarget < matriks[0].length) {
            
            int elemen = matriks[barisTarget][kolomTarget]; // Akses elemen
            System.out.println("Elemen pada baris " + barisTarget + ", kolom " + kolomTarget + " adalah: " + elemen);
        } else {
            System.out.println("Indeks baris atau kolom tidak valid.");
        }
    }
}"""
    },
    {
        "topic": "Array Multidimensi",
        "studi_kasus": "Menampilkan matriks identitas 3x3.",
        "tugas": "- Deklarasikan dan inisialisasi matriks 3x3\n- Gunakan loop untuk mengisi matriks identitas (1 pada diagonal, 0 lainnya)\n- Tampilkan matriks.",
        "code": """
public class MatriksIdentitas {
    public static void main(String[] args) {
        int ukuran = 3;
        int[][] matriks = new int[ukuran][ukuran];

        System.out.println("Matriks Identitas " + ukuran + "x" + ukuran + ":");
        for (int i = 0; i < ukuran; i++) {
            for (int j = 0; j < ukuran; j++) {
                if (i == j) { // Kondisi diagonal
                    matriks[i][j] = ...;  // Isi dengan 1 untuk diagonal
                } else {
                    matriks[i][j] = ...; // Isi dengan 0 untuk elemen non-diagonal
                }
                System.out.print(matriks[i][j] + " ");
            }
            System.out.println();
        }
    }
}"""
    },
    {
        "topic": "Array Multidimensi",
        "studi_kasus": "Menghitung rata-rata nilai setiap baris dalam matriks.",
        "tugas": "- Deklarasikan dan inisialisasi matriks 3x3 nilai siswa\n- Gunakan loop untuk menghitung total dan rata-rata setiap baris\n- Tampilkan rata-rata setiap baris.",
        "code": """
public class RataRataBarisMatriks {
    public static void main(String[] args) {
        int[][] nilaiSiswa = {
            {70, 80, 90},
            {65, 75, 85},
            {90, 95, 80}
        };

        System.out.println("Rata-rata nilai per baris:");
        for (int i = 0; i < nilaiSiswa.length; i++) {
            int totalBaris = 0;
            for (int j = 0; j < nilaiSiswa[i].length; j++) {
                totalBaris += nilaiSiswa[...][...]; // Tambahkan nilai elemen (i)(j)
            }
            double rataRata = (double) totalBaris / nilaiSiswa[i].length; // Hitung rata-rata
            System.out.println("Baris " + i + ": " + ...); // Tampilkan rata-rata (rataRata)
        }
    }
}"""
    },
    {
        "topic": "Fungsi Static",
        "studi_kasus": "Program kalkulator dengan fungsi-fungsi matematika dasar yang dapat dipanggil secara terpisah (static methods).",
        "tugas": "- Lengkapi return pada fungsi kurang dengan 'a - b'\n- Implementasikan return pada fungsi kali dengan 'a * b'\n- Panggil fungsi kurang dan bagi di main method",
        "code": """
public class KalkulatorFungsi {
    public static double tambah(double a, double b) {
        return a + b;
    }

    public static double kurang(double a, double b) {
        return ... - ...; // Operasi pengurangan (a - b)
    }

    public static double kali(double a, double b) {
        return ... * ...; // Operasi perkalian (a * b)
    }

    public static double bagi(double a, double b) {
        if(b == 0) {
            System.out.println("Error: Pembagian dengan nol!");
            return 0;
        }
        return a / b;
    }

    public static void main(String[] args) {
        double bilangan1 = 15.0;
        double bilangan2 = 3.0;

        System.out.println("=== KALKULATOR FUNGSI ===");
        System.out.println("Bilangan 1: " + bilangan1);
        System.out.println("Bilangan 2: " + bilangan2);
        System.out.println();
        
        System.out.println("Penjumlahan: " + tambah(bilangan1, bilangan2));
        System.out.println("Pengurangan: " + ...(bilangan1, bilangan2)); // Panggil fungsi kurang
        System.out.println("Perkalian: " + kali(bilangan1, bilangan2));
        System.out.println("Pembagian: " + ...(bilangan1, bilangan2)); // Panggil fungsi bagi
    }
}"""
    },
    {
        "topic": "Fungsi Static",
        "studi_kasus": "Membuat fungsi untuk menampilkan pesan selamat datang dengan nama yang diberikan.",
        "tugas": "- Buat fungsi static `sapaPengguna` yang menerima parameter String nama\n- Di dalam fungsi, cetak pesan 'Halo, [nama]! Selamat datang.'\n- Panggil fungsi ini di main method dengan nama 'Budi'.",
        "code": """
public class UcapanSelamatDatang {

    public static void sapaPengguna(String nama) { // Fungsi static
        System.out.println("Halo, " + ... + "! Selamat datang."); // Cetak pesan dengan parameter nama
    }

    public static void main(String[] args) {
        String namaPengguna = "..."; // Inisialisasi nama pengguna ("Budi")
        ...(namaPengguna); // Panggil fungsi sapaPengguna dengan nama pengguna
    }
}"""
    },
    {
        "topic": "Fungsi Static",
        "studi_kasus": "Membuat fungsi untuk menjumlahkan dua bilangan dan mengembalikan hasilnya.",
        "tugas": "- Buat fungsi static `jumlahkan` yang menerima dua parameter int a dan int b\n- Fungsi ini harus mengembalikan hasil penjumlahan a dan b\n- Panggil fungsi ini di main method dan tampilkan hasilnya.",
        "code": """
public class JumlahDuaBilangan {

    public static int jumlahkan(int a, int b) { // Fungsi untuk menjumlahkan
        return ... + ...; // Kembalikan hasil penjumlahan (a + b)
    }

    public static void main(String[] args) {
        int angka1 = 10;
        int angka2 = 20;
        int hasilPenjumlahan = ...(angka1, angka2); // Panggil fungsi jumlahkan

        System.out.println("Hasil penjumlahan " + angka1 + " dan " + angka2 + " adalah: " + hasilPenjumlahan);
    }
}"""
    },
    {
        "topic": "Fungsi Static",
        "studi_kasus": "Membuat fungsi untuk mencari nilai terbesar dari tiga angka.",
        "tugas": "- Buat fungsi static `cariTerbesar` yang menerima tiga parameter int\n- Fungsi ini harus mengembalikan nilai terbesar dari ketiga angka\n- Panggil fungsi ini di main method dan tampilkan hasilnya.",
        "code": """
public class CariNilaiTerbesar {

    public static int cariTerbesar(int a, int b, int c) { // Fungsi untuk mencari terbesar
        int max = a;
        if (b > max) {
            max = b;
        }
        if (c > max) {
            max = ...; // Set max ke c
        }
        return max;
    }

    public static void main(String[] args) {
        int num1 = 15;
        int num2 = 25;
        int num3 = 12;
        int terbesar = cariTerbesar(num1, num2, num3); // Panggil fungsi dengan num3
        System.out.println("Nilai terbesar adalah: " + terbesar);
    }
}"""
    },
    {
        "topic": "Fungsi Static",
        "studi_kasus": "Membuat fungsi untuk memeriksa apakah sebuah bilangan adalah prima atau tidak.",
        "tugas": "- Buat fungsi static `isPrima` yang menerima parameter int angka\n- Fungsi ini mengembalikan `true` jika angka prima, `false` jika tidak\n- Panggil fungsi ini di main method dan tampilkan hasilnya.",
        "code": """
public class CekPrimaFungsi {

    public static boolean isPrima(int angka) { // Fungsi isPrima
        if (angka <= 1) {
            return false;
        }
        for (int i = 2; i * i <= angka; i++) {
            if (angka % i == 0) {
                return false; // Jika bukan prima
            }
        }
        return true;
    }

    public static void main(String[] args) {
        int testAngka = 17;
        if (isPrima(testAngka)) { // Panggil fungsi isPrima
            System.out.println(testAngka + " adalah bilangan prima.");
        } else {
            System.out.println(testAngka + " bukan bilangan prima.");
        }
    }
}"""
    },
    {
        "topic": "Fungsi Rekursif",
        "studi_kasus": "Program untuk menghitung bilangan faktorial menggunakan pendekatan rekursif.",
        "tugas": "- Lengkapi kondisi basis rekursi dengan 'n <= 1'\n- Implementasikan panggilan rekursif dengan 'n * faktorial(n - 1)'\n- Set range untuk loop faktorial menjadi '5'\n- Set angka untuk test lebih besar menjadi '8'",
        "code": """
public class FaktorialRekursif {
    // Implementasikan fungsi rekursif untuk menghitung faktorial
    public static long faktorial(int n) {
        // Definisikan basis rekursi
        if(n <= 1) { // Kondisi basis rekursi
            return 1;
        }

        // Panggilan rekursif
        return n * faktorial(n - 1); // Panggilan rekursif
    }

    public static void main(String[] args) {
        System.out.println("=== PERHITUNGAN FAKTORIAL ===");

        // Hitung dan tampilkan faktorial untuk beberapa angka
        for(int i = 1; i <= 5; i++) { // Batas akhir loop (contoh: 5)
            long hasil = faktorial(i);
            System.out.println(i + "! = " + hasil);
        }
        
        // Test dengan angka yang lebih besar
        int angkaBesar = 8; // Bantuan 4: Masukkan angka (contoh: 8)
        System.out.println();
        System.out.println("Faktorial " + angkaBesar + " = " + faktorial(angkaBesar));
    }
}"""
    },
    {
        "topic": "Fungsi Rekursif",
        "studi_kasus": "Menghitung deret Fibonacci secara rekursif.",
        "tugas": "- Buat fungsi rekursif `fibonacci` yang mengembalikan nilai Fibonacci ke-n\n- Definisikan kondisi basis: jika n <= 1, kembalikan n\n- Panggil fungsi secara rekursif untuk n-1 dan n-2\n- Tampilkan deret Fibonacci hingga batas tertentu di main method.",
        "code": """
public class DeretFibonacciRekursif {

    public static int fibonacci(int n) {
        if (n <= ...) { // Kondisi basis (1)
            return n;
        }
        return fibonacci(n - 1) + fibonacci(n - ...); // Panggilan rekursif (n-2)
    }

    public static void main(String[] args) {
        int batas = ...; // Batas deret Fibonacci (8)
        System.out.println("Deret Fibonacci hingga ke-" + batas + ":");
        for (int i = 0; i < batas; i++) {
            System.out.print(fibonacci(i) + " ");
        }
        System.out.println();
    }
}"""
    },
    {
        "topic": "Fungsi Rekursif",
        "studi_kasus": "Mencari nilai maksimum dalam sebuah array menggunakan rekursi.",
        "tugas": "- Buat fungsi rekursif `cariMaksimum` yang menerima array, indeks awal, dan indeks akhir\n- Definisikan kondisi basis: jika indeks awal sama dengan indeks akhir, kembalikan elemen tersebut\n- Bandingkan elemen saat ini dengan maksimum dari sisa array secara rekursif\n- Panggil fungsi ini di main method dan tampilkan hasilnya.",
        "code": """
public class MaksimumArrayRekursif {

    public static int cariMaksimum(int[] arr, int indeksAwal, int indeksAkhir) {
        if (indeksAwal == indeksAkhir) { // Kondisi basis
            return arr[indeksAwal];
        }
        int maksSisa = cariMaksimum(arr, indeksAwal + 1, indeksAkhir);
        return Math.max(arr[indeksAwal], ...); // Bandingkan dengan maksSisa
    }

    public static void main(String[] args) {
        int[] data = {12, 5, 8, 25, 17, 3};
        int maksimum = cariMaksimum(data, 0, data.length - ...); // Indeks akhir (data.length - 1)
        System.out.println("Nilai maksimum dalam array adalah: " + maksimum);
    }
}"""
    },
    {
        "topic": "Fungsi Rekursif",
        "studi_kasus": "Menghitung jumlah digit suatu bilangan bulat secara rekursif.",
        "tugas": "- Buat fungsi rekursif `hitungDigit` yang menerima parameter int angka\n- Definisikan kondisi basis: jika angka < 10, kembalikan 1\n- Panggil fungsi secara rekursif untuk angka dibagi 10 dan tambahkan 1\n- Panggil fungsi ini di main method dan tampilkan hasilnya.",
        "code": """
public class HitungDigitRekursif {

    public static int hitungDigit(int angka) {
        if (angka < ...) { // Kondisi basis (10)
            return 1;
        }
        return 1 + hitungDigit(angka / ...); // Panggilan rekursif (angka / 10)
    }

    public static void main(String[] args) {
        int bilangan = 12345;
        int jumlah = ...(bilangan); // Panggil fungsi hitungDigit
        System.out.println("Jumlah digit dari " + bilangan + " adalah: " + jumlah);
    }
}"""
    },
    {
        "topic": "Fungsi Rekursif",
        "studi_kasus": "Mengecek apakah sebuah string adalah palindrom (dibaca sama dari depan dan belakang) secara rekursif.",
        "tugas": "- Buat fungsi rekursif `isPalindrom` yang menerima string, indeks awal, dan indeks akhir\n- Definisikan kondisi basis: jika indeks awal >= indeks akhir, kembalikan true\n- Bandingkan karakter di indeks awal dan akhir, lalu panggil rekursif untuk sisa string\n- Panggil fungsi ini di main method dan tampilkan hasilnya.",
        "code": """
public class PalindromRekursif {

    public static boolean isPalindrom(String str, int awal, int akhir) {
        if (awal >= akhir) { // Kondisi basis
            return true;
        }
        if (str.charAt(awal) != str.charAt(akhir)) {
            return false;
        }
        return isPalindrom(str, awal + 1, akhir - ...); // Panggilan rekursif (akhir - 1)
    }

    public static void main(String[] args) {
        String kata1 = "level";
        String kata2 = "hello";

        System.out.println(kata1 + " adalah palindrom: " + isPalindrom(kata1, 0, kata1.length() - ...)); // Indeks akhir (kata1.length() - 1)
        System.out.println(kata2 + " adalah palindrom: " + isPalindrom(kata2, 0, kata2.length() - 1));
    }
}"""
    },
]