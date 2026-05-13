# Tahap 2: Topik 5-8 (40 Soal Java Bervariasi)

EXPANDED_JAVA_EXAMPLES_P2 = [
    # ================= TOPIC 5: Perulangan While =================
    {
        "topic": "Perulangan While",
        "studi_kasus": "Simulasi pompa sumur hidrolik penguras banjir. Selama volume debit air kubangan masih lebih dari 0, mesin pompa tetap menyedot 50 liter per detik.",
        "tugas": "- Inisialisasi genanganAirL dengan 2500\n- Buat siklus While berjalan selagi volume air menggenangi area (> 0)\n- Kurangi dengan kecepatanSedotL per iterasi\n- Tampilkan log penyedotan",
        "code": """
public class PompaBanjir {
    public static void main(String[] args) {
        int genanganAirL = 2500;
        int kecepatanSedotL = 50;
        
        while (genanganAirL > ...) {
            genanganAirL = genanganAirL - ...;
            System.out.println("Sisa Debit Banjir: " + genanganAirL + " Liter");
        }
        System.out.println("Halaman rumah berhasil dikeringkan sepenuhnya!");
    }
}"""
    },
    {
        "topic": "Perulangan While",
        "studi_kasus": "Algoritma mesin kasino rolet otomatis memotong saldo chips pejudi. Pemain terus bertaruh Rp50.000 setiap putaran rolet selama sisa saldo modal minimal 50.000.",
        "tugas": "- While loop untuk memonitor sisa saldo judi\n- Tarik uang meja per putaran dalam loop",
        "code": """
public class KasinoRolet {
    public static void main(String[] args) {
        int saldoPemain = 150000;
        int spinCost = 50000;
        int totalSpin = 0;
        
        while (saldoPemain >= ...) {
            saldoPemain -= ...;
            totalSpin++;
            System.out.println("Putaran ke-" + totalSpin + ", Sisa Modal: Rp" + saldoPemain);
        }
        System.out.println("Bandar: Silakan isi ulang deposit Anda!");
    }
}"""
    },
    {
        "topic": "Perulangan While",
        "studi_kasus": "Enkripsi Brute-Force Password Cracker. Program jahat mencoba PIN dari 0000 terus bertambah satu angka hingga menyamai target kunci server (9999).",
        "tugas": "- Loop While hingga pinHacker sama persis dengan masterPinKey",
        "code": """
public class BruteForceLog {
    public static void main(String[] args) {
        int masterPinKey = 777;
        int pinHacker = 0;
        
        while (pinHacker != ...) {
            // Meretas algoritma gembok angka
            ...++;
        }
        System.out.println("Akses Administrator Terbobol pada pin = " + pinHacker);
    }
}"""
    },
    {
        "topic": "Perulangan While",
        "studi_kasus": "Pesawat komersial mengurangi ketinggian (Altitude) secara bertahap (500 kaki per menit) saat pendekatan landing hingga menyentuh landasan aspal darat (0 kaki).",
        "tugas": "- Iterasi While ketika altimeter di atas landasan (>= 0)\n- Minus altimeter",
        "code": """
public class PesawatLanding {
    public static void main(String[] args) {
        int ketinggianKaki = 8000;
        int sinkRate = 500;
        
        while (... > 0) {
            System.out.println("Tower: Ketinggian saat ini " + ketinggianKaki + " kaki.");
            ketinggianKaki = ketinggianKaki - ...;
        }
        System.out.println("Touchdown! Roda gigi masuk landasan mulus.");
    }
}"""
    },
    {
        "topic": "Perulangan While",
        "studi_kasus": "Mesin detektor radiasi nuklir Geiger counter terus berbunyi klakson pip jika angka radiasi gamma melebihi zona aman (< 30 mSv). Teknisi perlahan mendinginkan reaktor (Turun 5 mSv).",
        "tugas": "- Loop selama level gamma tidak aman.",
        "code": """
public class ReaktorNuklir {
    public static void main(String[] args) {
        int radiasiGamma = 100; // Kondisi genting awal
        
        while (radiasiGamma >= ...) {
            System.out.println("BEEP! BEEP! Radiasi " + radiasiGamma + " mSv.");
            radiasiGamma -= 5;
        }
        System.out.println("Radiasi Stabil di angka aman: " + radiasiGamma + " mSv.");
    }
}"""
    },
    {
        "topic": "Perulangan While",
        "studi_kasus": "Algoritma bot scalping crypto terus menjual saldo koin setiap detiknya sampai semua aset Kripto ETH yang ia pegang habis nol terjual di bursa spot.",
        "tugas": "- Perulangan While kuantitas koin Ethereum yang bisa dilempar per lot",
        "code": """
public class CryptoBot {
    public static void main(String[] args) {
        double walletEth = 2.5;
        double jualPerTik = 0.5;
        
        while (... > 0.0) {
            System.out.println("Eksekusi Jual ETH Pasar: Sisa " + walletEth);
            walletEth -= ...;
        }
        System.out.println("Aset terkuras total 0.");
    }
}"""
    },
    {
        "topic": "Perulangan While",
        "studi_kasus": "Dalam ekosistem biologi, bakteri Amuba akan terus membelah diri menjadi dua setiap iterasi waktu, dan berhenti jika populasi wadah kaca mikroskop melampaui limit (100.000).",
        "tugas": "- While koloni kurang dari limit 100K\n- Kalikan eksponensial dalam perputaran",
        "code": """
public class SelAmuba {
    public static void main(String[] args) {
        int jumlahAmuba = 100;
        int limitMaksimal = 100000;
        
        while (jumlahAmuba < ...) {
            System.out.println("Populasi: " + jumlahAmuba);
            jumlahAmuba = jumlahAmuba * ...;
        }
        System.out.println("Dish lab penuh! Bakteri stop membelah.");
    }
}"""
    },
    {
        "topic": "Perulangan While",
        "studi_kasus": "Satelit komunikasi merotasi posisinya di luar angkasa. Derajat kemiringan panel surya dikalibrasi naik 1 derajat secara bertahap agar sejajar menuju target 45 derajat arah kutub.",
        "tugas": "- while rotasi tidak sama dengan 45 derajat kemiringan target",
        "code": """
public class SatelitSurya {
    public static void main(String[] args) {
        int posisiDerajat = 15; // Meleset
        int targetDerajat = 45;
        
        while (posisiDerajat != ...) {
            posisiDerajat++;
            System.out.println("Memutar hidrolik satelit: " + posisiDerajat + " Derajat.");
        }
        System.out.println("Panel surya mengunci koordinat Sinar Matahari Penuh.");
    }
}"""
    },
    {
        "topic": "Perulangan While",
        "studi_kasus": "Manajemen memori sistem operasi: CPU menghapus baris queue cache antrean file (buffer) yang membengkak terus menerus sampai buffer menjadi bersih kosong kembali (0).",
        "tugas": "- Gunakan statement siklus membersihkan memory heap OS per fileblok",
        "code": """
public class CacheCleaner {
    public static void main(String[] args) {
        int queueBuffer = 55;
        
        while (queueBuffer > ...) {
            System.out.println("Menghapus index pointer lama cache os... Sisa: " + queueBuffer);
            queueBuffer--;
        }
        System.out.println("RAM Registry Sistem Kembali Lenggang.");
    }
}"""
    },
    {
        "topic": "Perulangan While",
        "studi_kasus": "Oven pabrik pengeringan keramik memanaskan ruang bakarnya. Suhu perlahan meningkat (10 Celcius per jam). Target adalah harus tepat mencapai 950 derajat tidak boleh lebih atau kurang.",
        "tugas": "- Lakukan while suhu < suhuTujuan\n- Add plus equals sepuluh variabel suhuBakar",
        "code": """
public class PabrikKeramik {
    public static void main(String[] args) {
        int suhuOven = 50; // Baru dinyalakan
        int targetSuhu = 950;
        
        while (... < ...) {
            suhuOven += 10;
        }
        System.out.println("Pemanggangan Keramik Tanah Liat Dimulai Suhu: " + suhuOven);
    }
}"""
    },

    # ================= TOPIC 6: Perulangan Do-While =================
    {
        "topic": "Perulangan Do-While",
        "studi_kasus": "Autentikasi brankas arsip gedung rahasia intelijen NSA. Karena model pintu analog, tuas akan selalu diputar minimal satu kali dulu sebelum sirkuit mengkonfirmasi apakah sandinya benar (Password=911).",
        "tugas": "- Lakukan proses autentikasi (Do blok awal).\n- Pastikan while loop menahan pintu sampai user input yang benar.",
        "code": """
public class BrankasNSA {
    public static void main(String[] args) {
        int sandiPinInput = 0;
        int kodeRahasia = 911;
        int tuasDiputar = 0;
        
        do {
            System.out.println("Menggeser tuas mekanik berbunyi klik...");
            tuasDiputar++; // Simulasikan tuas digerakkan fisik user
            if(tuasDiputar == 3) sandiPinInput = 911; // Di putaran ke-3, diasumsikan ia buka benar
            
        } while (sandiPinInput != ...);
        
        System.out.println("Engsel baja terlepas. Akses Dokumen Top-Secret.");
    }
}"""
    },
    {
        "topic": "Perulangan Do-While",
        "studi_kasus": "Menu interaktif dari Mesin ATM modern layar sentuh. Layar menu selalu ditampilkan minimum 1 kali kepada tiap nasabah yang baru saja memasukkan kartu debet sebelum nasabah memencet tombol Eject (0).",
        "tugas": "- Tampilkan layar do while\n- Kondisikan switch di dalam loop",
        "code": """
public class MenuATMLayar {
    public static void main(String[] args) {
        int pilihanLayar;
        int hitungSimulasi = 0;
        
        do {
            System.out.println("=== ATM CERDAS ===");
            System.out.println("1: Tarik Tunai");
            System.out.println("0: Ambil Kartu (Selesai)");
            
            hitungSimulasi++;
            pilihanLayar = (hitungSimulasi == 1) ? 1 : 0; // Iterasi pertama milih 1, baru keluar 0
            
            System.out.println("Anda memilih: " + pilihanLayar);
            
        } while (pilihanLayar != ...);
        
        System.out.println("Jangan lupa ambil kembali kartu cip debet Anda di lubang dispenser.");
    }
}"""
    },
    {
        "topic": "Perulangan Do-While",
        "studi_kasus": "Mesin ekskavator (Beko) pengeduk tambang memutar mangkuk bor ke permukaan aspal minimal satu kali tebasan, setelahnya sensor tanah keras (Rocky) akan memerintahkannya untuk menekan ulang atau berhenti (Rocky=True).",
        "tugas": "- Putar bor minimal satu kali di blok 'do'\n- Evaluasi material keras",
        "code": """
public class TambangEkskavator {
    public static void main(String[] args) {
        boolean batuBaraKeras = false;
        int meterGalian = 0;
        
        do {
            System.out.println("Menghantam aspal dengan gigi baja ekskavator!");
            meterGalian += 5;
            
            if(meterGalian >= 10) batuBaraKeras = true; // Simulasi mentok batu
            
        } while (... == false);
        
        System.out.println("Bor berhenti galian di kedalaman " + meterGalian + " Meter, Ujung mata terbentur logam mentah.");
    }
}"""
    },
    {
        "topic": "Perulangan Do-While",
        "studi_kasus": "Mobil otonom Tesla akan menyalakan wiper kaca depan menyapu hujan setidaknya satukali (Satu Usapan) begitu sensor pertama kali kejatuhan air sebelum memvalidasi intensitas siraman air.",
        "tugas": "- Eksekusi siklus do sweeping kaca\n- Cek variabel kondisi basah bool di akhir loop",
        "code": """
public class AutoWiperKaca {
    public static void main(String[] args) {
        boolean kacaBasah = true;
        int putaranUsap = 0;
        
        do {
            System.out.println("Sreeet! Sapuan karet karet Wiper dari Kanan Ke Kiri.");
            putaranUsap++;
            if (putaranUsap > 2) kacaBasah = false; // Kaca sudah bersih pasca 3 usapan
            
        } while (...);
        
        System.out.println("Sensor optikal membaca pandangan jernih.");
    }
}"""
    },
    {
        "topic": "Perulangan Do-While",
        "studi_kasus": "Pesan promosi (Spam Marketing Campaign) dikirim setidaknya satu kali SMS ke nomor operator HP Anda hari ini. Setelahnya, script mengecek apakah fitur Opt-Out (Unsubscribe) diajukan pengguna.",
        "tugas": "- Kirim Blast promo diskon pulsa dengan Do\n- Periksa variabel optOutStatus di kondisi pengecekan while",
        "code": """
public class SpamMarketingSms {
    public static void main(String[] args) {
        boolean pelangganOptOut = false;
        int hari = 1;
        
        do {
            System.out.println("Kirim SMS: Diskon Kuota Super Deal 50GB !! Hari ke- " + hari);
            hari++;
            if(hari == 3) pelangganOptOut = true; // User muak di hari ke 3 balasan OFF
            
        } while (!...);
        
        System.out.println("Nomor Anda Telah Dicabut Dari Daftar Blast Marketing Provider.");
    }
}"""
    },
    {
        "topic": "Perulangan Do-While",
        "studi_kasus": "Dalam olahraga balap Formula 1, pembalap harus mengaspal menjalani sesi lap kalibrasi pemanasan (Formation Lap) setidaknya 1 putaran mengelilingi sirkuit secara wajib sebelum lampu Race dimulai.",
        "tugas": "- Pastikan blok keliling ban berputar Do pertama\n- Eval lap counter balap keliling",
        "code": """
public class FormulaBanPanas {
    public static void main(String[] args) {
        int formasiLap = 0;
        int syaratSelesaiLap = 1;
        
        do {
            System.out.println("Pembalap zig-zag di tikungan DRS. Menghangatkan tapak ban Pirelli.");
            formasiLap++;
            
        } while (formasiLap < ...);
        
        System.out.println("GRID LOCK: Semua 20 Mobil Standby. LAMPU MERAH KELUAR FIVE LIGHTS GO!");
    }
}"""
    },
    {
        "topic": "Perulangan Do-While",
        "studi_kasus": "Program uji stres beban mesin paku tembak beton. Minimal pelatuk selalu bisa ditekan satu kali jepret (Piston bekerja), baru periksa sisa amunisi baut paku yang tersisa di magazin.",
        "tugas": "- Do tembakan paktu.\n- while sisaMagazine > 0 setelah iter",
        "code": """
public class PakuBetonBuilder {
    public static void main(String[] args) {
        int jumlahPaku = 3;
        
        do {
            System.out.println("DUARR! Paku menusuk tebal tembok beton cor!");
            ...--; // Turunkan magazine minus-minus
        } while (... > 0);
        
        System.out.println("Mesin kembang kempis paku kosong.");
    }
}"""
    },
    {
        "topic": "Perulangan Do-While",
        "studi_kasus": "Verifikasi face unlock HP minimal memindai infra merah satu kali flash wajah pengguna. Ia akan terus berkedip scanning jika data struktur hidung belum cocok 100%.",
        "tugas": "- Eksekusi Do flash infra merah\n- Variabel pencocokanWajah=True menghentikan loop.",
        "code": """
public class FaceUnlockID {
    public static void main(String[] args) {
        boolean mukaSamaDikenali = false;
        int detikScan = 0;
        
        do {
            System.out.println("Memancarkan array dot sensor Biometrik tulang pipi wajah...");
            detikScan++;
            if(detikScan >= 4) mukaSamaDikenali = true; // Simulasi nemu pas detik ke empat
            
        } while (... == false);
        
        System.out.println("Lockscreen Layar HP Terbuka Ke Halaman Utama (Home).");
    }
}"""
    },
    {
        "topic": "Perulangan Do-While",
        "studi_kasus": "Penyelidik medis memompa oksigen tabung ke selang paru-paru ventilator pasien Koma satu pompa manual dulu CPR, setelahnya denyut jantung (Ekg) dilihat untuk pompa lanjut atau stop.",
        "tugas": "- Lakukan CPR Do Block awal 1 pompa\n- Cek var denyutEkgDetensi",
        "code": """
public class JantungVentilator {
    public static void main(String[] args) {
        boolean ritmeEkgDitemukan = false;
        int jumlahSuntikanUdara = 0;
        
        do {
            System.out.println("Dokter menekan pompa kejut manual karet oksigen!!");
            jumlahSuntikanUdara++;
            if(jumlahSuntikanUdara > 2) ritmeEkgDitemukan = true; 
            
        } while (!...);
        
        System.out.println("Nadi pasien kembali berdegup! Pindahkan ke bangsal rawat inap biasa.");
    }
}"""
    },
    {
        "topic": "Perulangan Do-While",
        "studi_kasus": "Pelatihan jaringan saraf buatan (Neural Network / Machine Learning). Model AI melatih set data batch 1 Epoch utuh maju mundur, setibanya periksa jika skor akurasi (Error Loss) memuaskan.",
        "tugas": "- Do satu Epoch latih penuh neuron\n- Stop perulangan bila skor loss di bawah threshold kecil 0.05",
        "code": """
public class NeuralTrainingAI {
    public static void main(String[] args) {
        double skorErrorLoss = 0.99; // Sangat ngaco aslinya
        int epochModelIterasi = 0;
        
        do {
            epochModelIterasi++;
            System.out.println("Mesin Latih Gambar Epoch " + epochModelIterasi);
            skorErrorLoss -= 0.3; // Makin pintar dan turun errornya
            
        } while (... > 0.05);
        
        System.out.println("Semburan Cerdas! Akurasi Jaringan mencapai konvergensi final ideal Loss: " + skorErrorLoss);
    }
}"""
    },

    # ================= TOPIC 7: Sintaks Perulangan 1 (For Loops) =================
    {
        "topic": "Sintaks Perulangan 1",
        "studi_kasus": "Programmer membangun visual generator daftar barisan kursi penumpang Maskapai. Pesawat koridor tunggal (Narrow Body) dari baris ke-1 hingga titik kursi darurat baris ke-30 (Iterasi pasti).",
        "tugas": "- Pakai FOR Inisialisasi dari nomor 1 secara sekuensial nambah 1\n- Sampai titik akhir pembatas maksimal baris ke-30.",
        "code": """
public class PrintKursiPesawat {
    public static void main(String[] args) {
        int maxDudukLorong = 30;
        System.out.println("Mencetak Tiket Urutan Kabin Udara: ");
        
        for (int b = ...; b <= ...; ...) {
            System.out.println("Menyiapkan Sabuk Kursi Baris Penerbangan No - " + b);
        }
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 1",
        "studi_kasus": "Sistem otomatis rekayasa genetika menghitung deret kelipatan rantai DNA ganda (Double Helix). Mesin cetak RNA transkrip mencetak pola angka lompat loncat bilangan ganjil saja dari 1 hingga basa sel ke-21.",
        "tugas": "- Atur FOR berawal 1\n- Batasi kondisi stop sel 21\n- Incremental tidak plusplus tunggal tapi + 2 setiap loncatan rantai",
        "code": """
public class MesinGenetikDNA {
    public static void main(String[] args) {
        System.out.println("Menjalin Ikatan Transkripsi Rantai Basa Nitrogen Nukleat: ");
        
        for (int dnaGanjil = 1; ... <= 21; dnaGanjil += ...) {
            System.out.println("Memproduksi Sel Rantai Mutasi Kode Basa-" + dnaGanjil);
        }
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 1",
        "studi_kasus": "Aplikasi tracker pelacakan siklus kalender jam putaran gerhana bulan Blood Moon sekian milenial abad, mencetak tahun-tahun gerhana dengan ritme loncat konstan per 18 tahun (Siklus Saros) sejak tahun 2000 batas 2100.",
        "tugas": "- Inisialisasi awal for parameter dari dasar permulaan (2000)\n- Parameter pembatas limit abad penutup (2100)\n- Increment lompat interval matematis Astronomi gerhana periodik (+18)",
        "code": """
public class KalenderSiklusSaros {
    public static void main(String[] args) {
        System.out.println("Daftar Historis Astronomi Fenomena Bulan Ekstrem: ");
        
        for (int tahunBloodMoon = ...; tahunBloodMoon <= ...; tahunBloodMoon += ...) {
            System.out.println("Fase Gerhana Bulan Cincin Merah Darah (Penumbra) Pada " + tahunBloodMoon);
        }
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 1",
        "studi_kasus": "Papan billboard digital (Reklame Iklan Malam Angka LED). Menampilkan siklus animasi lampu huruf bergerak menurun mundur urut Countdown hitung detik perayaan Tahun Baru dari detik angka 10 terjun drastis ke 1 kembang api.",
        "tugas": "- Atur awal siklus decremental parameter inisialisasi detik=10\n- Kondisi sampai ambang pembatas > 0\n- Aksi siklus pengurangan secara konsisten decrementor",
        "code": """
public class ReklameTahunBaru {
    public static void main(String[] args) {
        System.out.println("Animasi LED Display Countdown Dimulai Pesta Confetti Jantung Kota!");
        
        for (int detik = ...; detik > ...; ...--) {
            System.out.println("LED BENTUK ANGKA DECREMENT TERANG: " + detik);
        }
        System.out.println("DUARRR! Kembang Api Mengudara Nyala Cerah!");
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 1",
        "studi_kasus": "Audit bank kriptografi block hash harian. Pemeriksa (Node Miner) memeriksa integritas catatan ledger block transakari dari nomor blok 500 sampai blok genesis awal 490 secara urutan mundur (Reverse iteration validation).",
        "tugas": "- Turunkan hitungan reverse FOR ledger block hash ID\n- Eksekusi dari 500 ke titik henti rentang batas bawah iter.",
        "code": """
public class LedgerAuditMiner {
    public static void main(String[] args) {
        System.out.println("Mengecek Integritas Rantai Validasi Hashing Terbalik: ");
        
        for (int blokHash = ...; blokHash >= 490; ...) {
            System.out.println("Validasi Ledger Kriptografi Kunci Block Merkle Ke-" + blokHash + " [TRUE - AMAN]");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 1",
        "studi_kasus": "Kamera pemantau drone agrikultur memotret baris hektaran ladang tebu secara linier lurus urut petak dari nomor sektor 1 sampail blok tanah sektor 12 untuk memeriksa penyakit kuning daun tumbuhan tebu secara seragam lurus.",
        "tugas": "- Majukan koordinat iterasi foto kamera UAV sektor ladang ke batas limit petakan",
        "code": """
public class DroneSurveiTebu {
    public static void main(String[] args) {
        System.out.println("Drone Mengaktifkan Mata Sensor Pencitraan Daun!");
        
        for (int blokPetakX = 1; blokPetakX <= ...; blokPetakX++ ) {
            System.out.println("Menyimpan Fotografi InfraMerah Plot Tebu Sektor Koordinat " + blokPetakX);
        }
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 1",
        "studi_kasus": "Robot las otomatis (Automated Robotic Arm) memoles sasis sambungan besi pelat mobil titik demi titik (Spot welding). Sesuai konfigurasi cetak biru pabrik (1 hingga posisi 8 sambungan baja tulang mobil linier).",
        "tugas": "- Nyalakan api busur panas per posisi FOR secara statis per list paku.",
        "code": """
public class RobotAsemblyLas {
    public static void main(String[] args) {
        int maxSpotBaja = 8;
        System.out.println("Booting Lengan Komponen Pneumatik Spot Listrik Cadas");
        
        for (int lasPos = ...; lasPos <= ...; ...++) {
            System.out.println("Crrrtt!! Membakar Titik Sasis Tulang Rangka ke-" + lasPos);
        }
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 1",
        "studi_kasus": "Simulasi pengereman ABS hidrolik mobil balap. ABS mencekik rem tidak secara langsung terkunci tancap, tapi berdenyut/pompa ritmis 15 kali putaran ganjalan rem per mili sekon sebelum memblok roda total anti selip meluncur.",
        "tugas": "- Perulangkan jumlah cetikan tekanan rem secara kontinu\n- Cetakan loop berjejal rentet dari 1 ke angka 15 pakem batas.",
        "code": """
public class ABSBrakesRally {
    public static void main(String[] args) {
        System.out.println("WASPADA TAHAN KECEPATAN! Injak Pedal Licin Hujan!!");
        
        for (int denyutKalifer = 1; denyutKalifer <= ...; denyutKalifer...) {
            System.out.println("Sentak Rem Jepitan Piringan Cakram Denyut Tekanan-" + denyutKalifer);
        }
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 1",
        "studi_kasus": "Sistem pendingin cairan (Watercooling) di mesin raksasa data center Google menghidupkan kipas putar bertahap gigi persneling RPM. Putaran meningkat konstan dalam satuan rentang FOR setiap kelipatan 1000 hingga mentok angin 5000 RPM putaran hisap maksimal.",
        "tugas": "- Susun RPM Incremental For loop dari awal\n- Increment variabel dengan nilai ribuan konstan",
        "code": """
public class FanCurveServer {
    public static void main(String[] args) {
        System.out.println("Peringatan Udara Panas Sensor Thermistor! Kipas RPM Naik Tangga: ");
        
        for (int fanRpm = 1000; fanRpm <= ...; fanRpm += ...) {
            System.out.println("Gemuruh Angin Radiator Kipas Sedotan Data Center " + fanRpm + " rpm/m!");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 1",
        "studi_kasus": "Fungsi cetak (Printer 3D Laser Sintering Nylon Tinta). Mengendapkan lapisan dasar filamen poliamida lapis demi lapis terukur milimeter (Layer) menyusun benda geometrik dari dasaran lapis 1 terendah ke lapisan atas paripurna pilar 50 bentuk final kubus plastiknya utuh kokoh vertikal.",
        "tugas": "- Iterate variabel tumpukan lelehan string printer material layer naik per lapisan batas Z parameter.",
        "code": """
public class ModelPrinterTigaDimensi {
    public static void main(String[] args) {
        System.out.println("Suhu Nozzle Extruder Tinta Plastik Cair Siap, Dimulai Dari Bed Bawah!");
        int totalLayerZAxis = 50;
        
        for (int cetakLayerH = 1; cetakLayerH <= ...; ...++) {
            System.out.println("Mengeringkan Endapan Filamen Lapisan Cetakan Penampang ke-" + cetakLayerH);
        }
        System.out.println("DING! Objek Plastik Kokoh Kubus 3D Selesai.");
    }
}"""
    },

    # ================= TOPIC 8: Sintaks Perulangan 2 (Nested Loops) =================
    {
        "topic": "Sintaks Perulangan 2",
        "studi_kasus": "Mesin Scanner medis MRI/CT-Scan Rumah Sakit mencetak gambaran sel biologis tulang berformat Grid irisan tipis otak/badan piksel-piksel berlapis. Outer membaca sumbu potong baris Horizontal anatomi irisan, inner loop melukis kotak piksel titik kedelaman Vertikal scan radioaktif.",
        "tugas": "- Inisiasi Nested For pembentuk matriks kolom dalam dari Baris Luar.\n- Cetak karakter grid piksel scan medikal padat rapat.",
        "code": """
public class MriCtScanPiksel {
    public static void main(String[] args) {
        int sumbuXBarisTubuh = 4;
        int pikselYKolomSel = 5;
        System.out.println("Memulai Penembakan Resonansi Magnetik Berlapis Tubuh Anatomi:");
        
        for (int bx = 1; bx <= ...; bx++) {
            for (int py = 1; py <= ...; py++) {
                System.out.print("[¤] "); // Menggambar kotak sel piksel MRI grid sel
            }
            System.out.println(); // Pindak baris ganti scan layer
        }
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 2",
        "studi_kasus": "Desainer antarmuka (UI) kursi Bioskop digital (Booking Teater XX1). Barisan bangku terurut abjad Alfabet untuk outer jajaran, sedang tempat duduk angka deret teater 1 hinga bangku kursi 10 ditaruh di for sarang lapisan dalam. Sistem mengkalkulasikan pola peta ruangan teater sinema 2 dimensi kursi penonton utuh blok persegi panjang teater.",
        "tugas": "- For luar barisan abjad alfabet A ke alfabet C\n- For dalam deret nomer urut 1 sampai kapasitas kolom 10 layar teater padat baris spasi.",
        "code": """
public class BioskopSeatMapMap {
    public static void main(String[] args) {
        int bangkuPerBarisLorong = 10;
        System.out.println("Denah Pesan Kursi Layar Merah Teater Blok Film: ");
        
        for (char urutAbjadA = 'A'; urutAbjadA <= ...; urutAbjadA++) {
            for (int jokNomer = 1; jokNomer <= ...; jokNomer++) {
                System.out.print(urutAbjadA + "-" + jokNomer + " ");
            }
            System.out.println(); // Lompat barisan baru Lorong turun ke belakang
        }
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 2",
        "studi_kasus": "Radar navigasi kapal pesiar bahari memindai laut kordinat Lat-Lon satelit bujur lintang. Satelit menyisir kuadran area Samudera dengan pola kisi-kisi (Grid search). Loop luas X luar (Lintang Bujur) dipanggil di atas loop detak Y dalam peta pemetaan laut (titik Sonar kedalaman gelombang gelisah samudera biru pemetaan ikan karang).",
        "tugas": "- Buat kondisi inner loop koordinat area sarang.\n- Mengulang For ganda grid pelacak objek.",
        "code": """
public class RadarSonarSamudera {
    public static void main(String[] args) {
        int zonaBujurSearchX = 3;
        int zonaLintangSearchY = 4;
        System.out.println("Satelit Gelombang Bawah Laut Memisir Zona Pemetaan Misteri:");
        
        for (int kuadranB = 1; kuadranB <= ...; kuadranB++) {
            System.out.print("Sektor Lintasan Bujur " + kuadranB + " -> Memecah titik: ");
            for (int lautLi = 1; lautLi <= ...; lautLi++) {
                System.out.print("<T" + lautLi + "> ");
            }
            System.out.println();
        }
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 2",
        "studi_kasus": "Generator kunci (Keygen) sistem sandi mesin rahasia militer Enigma memutar kombinasi 2 roda mekanik pin bergerigi angka sandi cypher disk kunci. Roda roda utama outer diputar, selama per satu gigi klik perputaran outer gir baja, gigi gir gembok kombinasi dalam berputar sekian penuh digit kunci (Permutasi Nested rotasi gir putaran 3 level poros mesin kuno Perang Dunia sandi gembok kode mesin gerigi kuningan).",
        "tugas": "- Atur for berantai gir dalam untuk memutar poros gerigi dari kunci luar roda mesin.",
        "code": """
public class MesinSandiGirLogam {
    public static void main(String[] args) {
        int putaranPorosOuterGirPin = 3;
        int putaranPinInnerSubGir = 3;
        
        for (int girGembokA = 1; girGembokA <= ...; girGembokA++) {
            for (int subGigiB = 1; subGigiB <= ...; subGigiB++) {
                System.out.print("Mencoba Kombinasi Pin Kawat: " + girGembokA + "-" + subGigiB);
                System.out.print("  [Cekak, Kres]  ");
            }
            System.out.println(); // Pindah poros rel slot logam sandi
        }
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 2",
        "studi_kasus": "Dinding pembatas matriks sel surya di hamparan padang sahara disusun baris deret panel. Program harus menghitung bayangan tumpukan pelat solar panel kaca, secara piramida segi tegak terbalik susunan array kaca untuk mendeteksi terik bayangan menutupi lapisan array sudut barisan for loop bersusun menurun miring piramid cahaya panel atap menurun lereng lahan fotovoltaik susunan.",
        "tugas": "- Inisialisasi Nested Pattern pyramid mencetak miring blok.\n- Iterasi For kedua tergantung bergantung iterasi I di loop pertama per penampang sel pancaran sisi surya atap silikon miring teratur (Pola miring).",
        "code": """
public class PiramidaPanelSurya {
    public static void main(String[] args) {
        int tingkatPiramidaTinggi = 5;
        System.out.println("Konstruksi Pundak Terasering Kaca Silikon Radiasi Turun:");
        
        for (int tierTerasLapisDalam = 1; tierTerasLapisDalam <= ...; tierTerasLapisDalam++) {
            for (int sambungKacaD = 1; sambungKacaD <= ...; sambungKacaD++) {
                System.out.print("■"); // Kotak Solar Kaca gelap
            }
            System.out.println();
        }
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 2",
        "studi_kasus": "Membangun papan permainan Catur/Dam catur kuno 8x8 piksel terminal cetak hitam-putih. For luar (Baris Papan 8 Y) dipadu For dalam (Kolom X letak selak-seling pion papan). Operasi if ganjil-genap mod selang-seling blok menentukan cetakan persegi gelap atau blok bata warna terang susunan beruntun blok warna bidak permainan.",
        "tugas": "- Barisan outer ubin catur For pertama kotak baris Catur.\n- Petak for selang seling Catur papan perulangan dalam diisi statement bidak IF bersarang pencetak blok padat tebal tebal papan matriks checkerboard game logika papan adu strategik klasik catur warna berundak 8 kali 8 petakata grid sel catur warna kombinasi hitam putih biner.",
        "code": """
public class PapanPionCheckerGame {
    public static void main(String[] args) {
        int barisCaturAtasBawah = 8;
        int kolomCaturKiriKanan = 8;
        
        for (int papanY = 1; papanY <= ...; papanY++) {
            for (int selX = 1; selX <= ...; selX++) {
                if ((papanY + selX) % 2 == 0) {
                    System.out.print("██"); // Ubin papan warna coklat gelap / tertutup
                } else {
                    System.out.print("  "); // Ubin papan putih kosongan warna / terang catur
                }
            }
            System.out.println(); // Geser ubin cetaknya
        }
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 2",
        "studi_kasus": "Jam tangan Digital Pintar kronograf waktu berjalan detik sinkron dengan menit rotasi kronometer (Nested loops alamiah waktu). Detik berputar (inner loop) secara penuh 0 hingga kecepatan 59 satuan batas per sekon, baru kemudian per satu lingkaran tuntas penuh, sang Menit (outer loop indikator utama angka muka) mencentang naik jarumnya berdetak satu increment membesarkan layar menit jam atom digital mekanik display jarum kronos ganda timer jam sirkuit sekon mikro chip kristal putaran taktis waktu konstan roda gigi loop waktu.",
        "tugas": "- Perulangan luar kontrol Menit indikator display jam dasar 0 sampai 2 max.\n- Perulangan dalam cetak hitungan sekon detik angka mundur per 5 batas ringkasan timer putaran 0 batas satuan cepat lompat detik indikator cepat batas waktu.",
        "code": """
public class JamKronometerBiner {
    public static void main(String[] args) {
        int targetMenitHabis = 2; // Hanya 2 Menit tes limit alarm
        int cincinSiklusDetik = 5; // Detik cuma 5 sekon ilustrasi kompres ringkas cepat
        
        System.out.println("Memulai Detak Digital Jam Tangan Nuklir Chrono:");
        for (int m_Menit = 0; m_Menit < ...; m_Menit++) {
            for (int s_sekonDetik = 0; s_sekonDetik < ...; s_sekonDetik++) {
                System.out.println(String.format("Waktu Digital -> %02d : %02d", m_Menit, s_sekonDetik));
            }
        }
        System.out.println("TET TOT ALARM BERBUNYI CINCIN JAM HITUNG HABIS");
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 2",
        "studi_kasus": "Dalam arsitektur konstruksi gedung pencakar langit baja beton (Skyscraper Tower block bangunan struktur balok tumpuk tinggi). Alat derek Tower Crane menyusun tingkat blok apartemen (Outer loop lantai tingkat) secara menumpuk naik ke angkasa. Per lantai hunian dalam, crane mengecor tiang koridor jajaran pilar beton sekat ruangan lantai horizontal datar struktur apartement for bersarang lantai cor vertikal baris flat rusun tingkat grid jendela.",
        "tugas": "- For luar struktur tiang tumpuk tingkat bangunan naik.\n- For sarang kamar barisan hunian pintu sekat kamar nomor flat sejajar tiap per lantainya berjejer rapat mendatar cor bata horisontal sel pilar baja tingkat susunan beton flat kubikal apartemen petak.",
        "code": """
public class BangunTowerCrane {
    public static void main(String[] args) {
        int letakLantaiHunian = 6;
        int kamarUnitLebar = 4;
        
        for (int l_lantaiKe = letakLantaiHunian; l_lantaiKe >= ...; l_lantaiKe--) {
            System.out.print("Lantai Atas Ke [" + l_lantaiKe + "] -> ");
            for (int k_PintuSebelah = 1; k_PintuSebelah <= ...; k_PintuSebelah++) {
                System.out.print("Pintu_" + l_lantaiKe + "0" + k_PintuSebelah + " | ");
            }
            System.out.println();
        }
        System.out.println("================================ PONDASI DASAR TANAH KELAR");
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 2",
        "studi_kasus": "Pola serangan roket meriam pertahanan (Air Defense Artileri Peluru Rudal peluncuran rokok per meriam lintasan). Pertahanan pangkalan angkatan laut memiliki 3 Pos Senjata Turret Utama meriam balistik canon. Setiap turret meriam menembakan serentetan (Burst) beruntun 5 letupan peluru logam peledak berapi penangkis serangan udara secara berjejer per Pos pertahanan ganda For rentet loop senjata turret rentetan barikade serang laras mesiu api menembak langit For ledakan lapis tembakan rudal laras peluncur selang rotasi barel artileri salvo misil menembak musuh peluru terbang.",
        "tugas": "- For luar barisan mesin Turret meriam pertahanan.\n- Memanggil For dalam burst letupan tembakan laras rotasi roket misil meriam rentetan barel senapan letusan berapi rudal pos per stasiun salvo tembakan rudal serangan angkasa.",
        "code": """
public class PertahananRudalUdara {
    public static void main(String[] args) {
        int posTurretMeriam = 3;
        int tembakanBurstPeluru = 5;
        
        for (int kanonT = 1; kanonT <= ...; kanonT++) {
            System.out.print("Pos Turret Anti Udara-" + kanonT + " Menembak: ");
            for (int peluruP = 1; peluruP <= ...; peluruP++) {
                System.out.print("»DOR! "); // Tembakan ke angkasa melesat
            }
            System.out.println(" (Asap Kepul Meriam Barel Berputar Habis Reload Amunisi)");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Perulangan 2",
        "studi_kasus": "Peternak otomatis pemerah ayam kandang tingkat baterai peternakan telur industri baterik kandang (Factory Farm). Barisan lorong susunan tingkatan kandang (Blok Aatas ke Bawah bertingkat 3 level outer). Di dalam masing blok rak, terdapat 6 kurungan ayam individu di sisi deret dalam layer dalam lorong ayam (Nested memerah kandang mengambil kumpulkan kotak telur sebutir lorong level peternakan iterasi blok bilik sangkar telur jaring kandang kandang kawat).",
        "tugas": "- Lapisan luar untuk tumpukan laci tingkat Rak susun lorong kandang baterai.\n- Inner lorong for sisiran ayam unggas untuk di panen telur per lorong titik loker kawat besi peternakan panen produksi for sisir telur kandang keranjang iterasi panen hasil berlimpah per level ayam baterai dalam kotak sangkarnya berbaris kandang iteratif.",
        "code": """
public class PanenAyamKandang {
    public static void main(String[] args) {
        int safBawahAtasKandang = 3;
        int jejerSekatSangkar = 6;
        System.out.println("Operator Mesin Baterai Konveyor Berjalan Panen Produksi Telur Telur:");
        
        for (int rakLapisR = 1; rakLapisR <= ...; rakLapisR++) {
            System.out.print("Tingkat Besi Kawat Ke-" + rakLapisR + " [");
            for (int ayamA = 1; ayamA <= ...; ayamA++) {
                System.out.print(" (o) "); // Simbol telur kawat putih di conveyor belt panen berguling telur per baris
            }
            System.out.println("] Masuk Keranjang.");
        }
    }
}"""
    }
]
